"""
Class which manages a Pantograph instance. All calls to the kernel uses this
interface.
"""
import json, unittest, os, asyncio, tempfile
from typing import Union, List, Optional, Dict, List, Any, Tuple
from pathlib import Path

from pantograph.message import (
    Position,
    Severity,
    Message,
    TacticFailure,
    ParseError,
    ServerError,
)
from pantograph.expr import (
    parse_expr,
    Expr,
    Variable,
    Goal,
    GoalState,
    Subsumption,
    Site,
    Tactic,
    TacticHave,
    TacticLet,
    TacticMode,
    TacticExpr,
    TacticDraft,
)
from pantograph.utils import (
    to_sync,
    _get_proc_cwd,
    _get_proc_path,
    get_lean_path_async,
    get_lean_path,
)
from pantograph.data import (
    CheckTrackResult,
    CompilationUnit,
    SearchTarget,
)


class Server:
    """
    Main interaction instance with Pantograph.

    Asynchronous and synchronous versions are provided for each function.
    """

    def __init__(
            self,
            imports: List[str]=["Init"],
            project_path: Optional[str]=None,
            lean_path: Optional[str]=None,
            # Options for executing the REPL.
            options: Dict[str, Any]={},
            # Options supplied to the Lean core
            core_options: List[str]=[],
            timeout: int=60,
            buffer_limit: Optional[int]=1000000,
            _sync_init: bool=True):
        """
        options: Given to Pantograph
        core_options: Given to Lean core
        timeout: Amount of time to wait for execution (in seconds)
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.project_path = project_path if project_path else _get_proc_cwd()
        if _sync_init and project_path and not lean_path:
            lean_path = get_lean_path(project_path)
        self.lean_path = lean_path
        self.buffer_limit = buffer_limit
        self.proc_path = _get_proc_path()

        self.options = options
        self.core_options = core_options
        self.args = imports + [f'--{opt}' for opt in core_options]
        self.proc = None
        if _sync_init:
            self.restart()

        # List of goal states that should be garbage collected
        self.to_remove_goal_states = []

    @classmethod
    async def create(
            cls,
            imports: List[str]=["Init"],
            project_path: Optional[str]=None,
            lean_path: Optional[str]=None,
            # Options for executing the REPL.
            # Set `{ "automaticMode" : False }` to handle resumption by yourself.
            options: Dict[str, Any]={},
            core_options: List[str]=[],
            timeout: int=120,
            buffer_limit: Optional[int]=1000000,
            start: bool=True) -> 'Server':
        """
        timeout: Amount of time to wait for execution (in seconds)
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self = cls(
            imports,
            project_path,
            lean_path,
            options,
            core_options,
            timeout,
            buffer_limit,
            _sync_init=False
        )
        if project_path and not lean_path:
            lean_path = await get_lean_path_async(project_path)
        self.lean_path = lean_path
        if start:
            await self.restart_async()
        return self

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._close()

    def __del__(self):
        pass #self._close()

    def _close(self):
        if not self.proc:
            return

        self.proc.terminate()
        self.proc = None

    def is_automatic(self):
        """
        Check if the server is running in automatic mode
        """
        return self.options.get("automaticMode", True)

    async def restart_async(self):
        """
        Restart the server
        """
        self._close()
        env = os.environ
        if self.lean_path:
            env = env | {'LEAN_PATH': self.lean_path}

        self.proc = await asyncio.create_subprocess_exec(
            self.proc_path,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            #stderr=asyncio.subprocess.PIPE,
            cwd=self.project_path,
            env=env,
            limit=self.buffer_limit,
        )
        await self.proc.stdin.drain()
        try:
            ready = await asyncio.wait_for(self.proc.stdout.readline(), self.timeout)
            ready = ready.decode().strip()
            assert ready == "ready.", f"Server failed to emit ready signal: {ready}; This could be caused by Lean version mismatch between the project and Pantograph or insufficient timeout."
        except asyncio.TimeoutError as ex:
            raise RuntimeError("Server failed to emit ready signal in time") from ex

        if self.options:
            await self.run_async("options.set", self.options)

    restart = to_sync(restart_async)

    async def run_async(self, cmd, payload):
        """
        Runs a raw JSON command. Preferably use one of the commands below.

        :meta private:
        """
        assert self.proc, "Server not running."

        s = json.dumps(payload, ensure_ascii=False)
        command = f"{cmd} {s}\n"
        self.proc.stdin.write(command.encode())
        await self.proc.stdin.drain()
        line = ""
        try:
            line = await asyncio.wait_for(self.proc.stdout.readline(), self.timeout)
        except asyncio.TimeoutError as e:
            self._close()
            raise ServerError("Server reached timeout limit") from e

        try:
            line = line.decode().strip()
            return json.loads(line)
        except UnicodeDecodeError as e:
            self._close()
            raise ServerError(f"Could not decode process output: {line}") from e
        except json.JSONDecodeError as e:
            self._close()
            raise ServerError(f"Cannot decode Json object from: {line}") from e

    run = to_sync(run_async)

    async def gc_async(self):
        """
        Garbage collect deleted goal states to free up memory.
        """
        if not self.to_remove_goal_states:
            return
        result = await self.run_async('goal.delete', {'stateIds': self.to_remove_goal_states})
        self.to_remove_goal_states.clear()
        if "error" in result:
            raise ServerError(result)

    gc = to_sync(gc_async)

    async def expr_type_async(self, expr: Expr) -> Expr:
        """
        Evaluate the type of a given expression. This gives an error if the
        input `expr` is ill-formed.
        """
        result = await self.run_async('expr.echo', {"expr": expr})
        if "error" in result:
            raise ServerError(result)
        return parse_expr(result["type"])

    expr_type = to_sync(expr_type_async)

    async def goal_start_async(self, expr: Expr) -> GoalState:
        """
        Create a goal state with one root goal, whose target is `expr`
        """
        result = await self.run_async('goal.start', {"expr": str(expr)})
        if "error" in result:
            print(f"Cannot start goal: {expr}")
            raise ServerError(result)
        return GoalState(
            state_id=result["stateId"],
            goals=[Goal.sentence(expr)],
            messages=[],
            _sentinel=self.to_remove_goal_states,
        )

    goal_start = to_sync(goal_start_async)

    async def goal_root_async(self, state: GoalState) -> Optional[Expr]:
        """
        Print the root expression of a goal state
        """
        args = {"stateId": state.state_id, "rootExpr": True}
        result = await self.run_async('goal.print', args)
        if "error" in result:
            raise ServerError(result)
        root = result.get('root')
        if root is None:
            return None
        return parse_expr(root)

    goal_root = to_sync(goal_root_async)

    async def goal_tactic_async(self, state: GoalState, tactic: Tactic, site: Site = Site()) -> GoalState:
        """
        Execute a tactic on `goal_id` of `state`
        """
        args = {"stateId": state.state_id, **site.serial()}
        match tactic:
            case str():
                args["tactic"] = tactic
            case TacticHave():
                args["have"] = tactic.branch
                if tactic.binder_name:
                    args["binderName"] = tactic.binder_name
            case TacticLet():
                args["let"] = tactic.branch
                if tactic.binder_name:
                    args["binderName"] = tactic.binder_name
            case TacticExpr():
                args["expr"] = tactic.expr
            case TacticDraft():
                args["draft"] = tactic.expr
            case TacticMode():
                args["mode"] = tactic.serial()
            case _:
                raise RuntimeError(f"Invalid tactic type: {type(tactic)}")
        result = await self.run_async('goal.tactic', args)
        next_state_id = result.get("nextStateId")
        if "error" in result:
            raise ServerError(result)
        if "parseError" in result:
            raise TacticFailure(result)

        messages = result.get("messages")
        if "goals" not in result:
            raise TacticFailure([Message.parse(m) for m in messages])

        if result["hasSorry"]:
            await self.run_async('goal.delete', {'stateIds': [next_state_id]})
            raise TacticFailure("Tactic generated sorry", messages)
        if result["hasUnsafe"]:
            await self.run_async('goal.delete', {'stateIds': [next_state_id]})
            raise TacticFailure("Tactic generated unsafe", messages)

        return GoalState.parse(result, messages, self.to_remove_goal_states)

    goal_tactic = to_sync(goal_tactic_async)

    async def goal_continue_async(self, target: GoalState, branch: GoalState) -> GoalState:
        """
        After finish searching `target`, resume search on `branch`
        """
        result = await self.run_async('goal.continue', {
            "target": target.state_id,
            "branch": branch.state_id,
        })
        if "error" in result:
            raise ServerError(result)
        if "tacticErrors" in result:
            raise ServerError(result)
        if "parseError" in result:
            raise ServerError(result)
        return GoalState.parse(result, [], self.to_remove_goal_states)

    goal_continue = to_sync(goal_continue_async)

    async def goal_resume_async(self, state: GoalState, goals: list[Goal]) -> GoalState:
        """
        Bring `goals` back into scope
        """
        result = await self.run_async('goal.continue', {
            "target": state.state_id,
            "goals": [goal.name for goal in goals],
        })
        if "error" in result:
            raise ServerError(result)
        if "tacticErrors" in result:
            raise ServerError(result)
        if "parseError" in result:
            raise ServerError(result)
        return GoalState.parse(result, [], self.to_remove_goal_states)
    goal_resume = to_sync(goal_resume_async)

    async def goal_subsume_async(self, state: GoalState, goal: Goal, srcs: list[Goal]) -> (Optional[GoalState], Subsumption):
        """
        [Experimental] Detect subsumption
        """
        result = await self.run_async('goal.subsume', {
            "stateId": state.state_id,
            "goal": goal.id,
            "srcs": [g.id for g in srcs],
        })
        if "error" in result:
            raise ServerError(result)
        nextState = None
        if state_id := result.get("stateId"):
            nextState = GoalState(
                state_id=state_id,
                goals=[g for g in state.goals if g.id != goal.id],
                messages=[],
                _sentinel=self.to_remove_goal_states,
            )
        sub = Subsumption[result["result"].upper()]
        return (nextState, sub)

    goal_subsume = to_sync(goal_subsume_async)

    async def env_add_async(
            self, name: str, levels: list[str],
            t: Expr, v: Expr, is_theorem: bool = True):
        """
        Adds a definition to the environment.

        NOTE: May have to accept additional parameters if the definition
        contains universe mvars.
        """
        result = await self.run_async('env.add', {
            "name": name,
            "levels": levels,
            "type": t,
            "value": v,
            "isTheorem": is_theorem,
            "typeErrorsAsGoals": False,
        })
        if "error" in result:
            raise ServerError(result["desc"])

    env_add = to_sync(env_add_async)

    async def env_inspect_async(
            self,
            name: str,
            print_value: bool = False,
            print_dependency: bool = False) -> Dict:
        """
        Print the type and dependencies of a constant.
        """
        result = await self.run_async('env.inspect', {
            "name": name,
            "value": print_value,
            "dependency": print_dependency,
            "source": True,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        return result
    env_inspect = to_sync(env_inspect_async)

    async def env_module_read_async(self, module: str) -> dict:
        """
        Reads the content from one Lean module including what constants are in
        it.
        """
        result = await self.run_async('env.module_read', {
            "module": module
        })
        if "error" in result:
            raise ServerError(result["desc"])
        return result
    env_module_read = to_sync(env_module_read_async)

    async def env_parse_async(self, src: str, category: str = "tactic") -> Tuple[str, str]:
        """
        Parse an input using a syntax category's parser. Returns the parsed
        component and the tail.
        """
        result = await self.run_async('env.parse', {
            "input": src,
            "category": category,
        })
        if "error" in result:
            if result['error'] == 'parse':
                raise ParseError(result["desc"])
            raise ServerError(result["desc"])
        pos = result["pos"]
        s = src.encode()
        return s[:pos].decode(), s[pos:].decode()

    env_parse = to_sync(env_parse_async)

    async def env_save_async(self, path: str):
        """
        Save the current environment to a file
        """
        result = await self.run_async('env.save', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])
    env_save = to_sync(env_save_async)

    async def env_load_async(self, path: str):
        """
        Load the current environment from a file
        """
        result = await self.run_async('env.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])

    env_load = to_sync(env_load_async)

    async def goal_save_async(self, goal_state: GoalState, path: str):
        """
        Save a goal state to a file
        """
        result = await self.run_async('goal.save', {
            "id": goal_state.state_id,
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])

    goal_save = to_sync(goal_save_async)

    async def goal_load_async(self, path: str) -> GoalState:
        """
        Load a goal state from a file.

        User is responsible for keeping track of the environment.
        """
        result = await self.run_async('goal.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        state_id = result['id']
        result = await self.run_async('goal.print', {
            'stateId': state_id,
            'goals': True,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        return GoalState.parse_inner(
            state_id,
            result['goals'], [],
            self.to_remove_goal_states,
        )

    goal_load = to_sync(goal_load_async)

    async def tactic_invocations_async(self, file_name: Union[str, Path]) -> List[CompilationUnit]:
        """
        Collect tactic invocation points in file, and return them.
        """
        with tempfile.TemporaryDirectory() as tempdirname:
            invocation_file_name = f"{tempdirname}/invocations.json"
            result = await self.run_async('frontend.process', {
                'fileName': str(file_name),
                'invocations': invocation_file_name,
                "readHeader": True,
                "inheritEnv": False,
                "newConstants": False,
            })
            if "error" in result:
                raise ServerError(result)

            with open(invocation_file_name, "r") as f:
                data_units = json.load(f)
                units = [
                    CompilationUnit.parse(payload, invocations=data_unit["invocations"])
                    for payload, data_unit in zip(result['units'], data_units['units'])
                ]
                return units

    tactic_invocations = to_sync(tactic_invocations_async)

    async def load_header_async(self, header: str):
        """
        Loads the environment from a header. Set `imports` to `[]` during
        server creation to use this function.
        """
        result = await self.run_async('frontend.process', {
            'file': header,
            "newConstants": False,
            "readHeader": True,
            "inheritEnv": True,
        })
        if "error" in result:
            raise ServerError(result)

    load_header = to_sync(load_header_async)

    async def load_definitions_async(self, snippet: str):
        """
        Loads definitions in some Lean code and update the environment.

        Existing goal states will not automatically inherit said definitions.
        """
        result = await self.run_async('frontend.process', {
            'file': snippet,
            "newConstants": False,
            "readHeader": False,
            "inheritEnv": True,
        })
        if "error" in result:
            raise ServerError(result)

    load_definitions = to_sync(load_definitions_async)

    async def check_compile_async(
            self,
            code: str,
            new_constants: bool = False,
            read_header: bool = False):
        """
        Check if some Lean code compiles
        """
        result = await self.run_async('frontend.process', {
            'file': code,
            "newConstants": new_constants,
            "readHeader": read_header,
            "inheritEnv": False,
        })
        if "error" in result:
            raise ServerError(result)
        units = [
            CompilationUnit.parse(payload, goal_state_sentinel=self.to_remove_goal_states)
            for payload in result['units']
        ]
        return units

    check_compile = to_sync(check_compile_async)

    async def load_sorry_async(
            self,
            src: str,
            binder_name: Optional[str] = None,
            ignore_values: bool = True) -> list[SearchTarget]:
        """
        Condense search target into goals
        """
        args = {"file": src, "ignoreValues": ignore_values}
        if binder_name is not None:
            args["binderName"] = binder_name
        result = await self.run_async('frontend.distil', args)
        if "error" in result:
            raise ServerError(result)
        units = [
            SearchTarget.parse(payload, goal_state_sentinel=self.to_remove_goal_states)
            for payload in result['targets']
        ]
        return units

    load_sorry = to_sync(load_sorry_async)

    async def check_track_async(self, src: str, dst: str) -> CheckTrackResult:
        """
        Checks if `dst` file conforms to the specifications in `src`
        """
        result = await self.run_async('frontend.track', {"src": src, "dst": dst})
        if "error" in result:
            raise ServerError(result)
        src_messages = [Message.parse(d) for d in result["srcMessages"]]
        dst_messages = [Message.parse(d) for d in result["dstMessages"]]
        return CheckTrackResult(
            src_messages,
            dst_messages,
            failure=result.get("failure"),
        )

    check_track = to_sync(check_track_async)

    async def refactor_search_target_async(
            self,
            code: str,
            core_options: list[str] = []) -> str:
        """
        Combine multiple `sorry`s into one `sorry` using subtyping. It only
        supports flat dependency structures.

        This feature is experimental and depends on the round-trip capabilities
        of the delaborator.
        """
        result = await self.run_async('frontend.refactor', {
            'file': code,
            'coreOptions': core_options,
        })
        if "error" in result:
            raise ServerError(result)
        return result["file"]

    refactor_search_target = to_sync(refactor_search_target_async)

def get_version() -> str:
    """
    Returns the current Pantograph version for diagnostics purposes.
    """
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"],
                          stdout=subprocess.PIPE,
                          cwd=_get_proc_cwd()) as p:
        return p.communicate()[0].decode('utf-8').strip()


class TestServer(unittest.TestCase):

    def test_version(self):
        """
        NOTE: Update this after upstream updates.
        """
        self.assertEqual(get_version(), "0.3.7/v4.21.0")

    def test_server_init_del(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            server = Server()
            server.expr_type("forall (n m: Nat), n + m = m + n")
            del server
            server = Server()
            server.expr_type("forall (n m: Nat), n + m = m + n")
            del server
            server = Server()
            server.expr_type("forall (n m: Nat), n + m = m + n")
            del server

    def test_expr_type(self):
        server = Server()
        t = server.expr_type("forall (n m: Nat), n + m = m + n")
        self.assertEqual(t, "Prop")

    def test_goal_start(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, tactic="intro a")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            "_uniq.11",
            variables=[Variable(name="a", t="Prop")],
            target="∀ (q : Prop), a ∨ q → q ∨ a",
            name=None,
        )])
        self.assertEqual(str(state1.goals[0]),"a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a")

        del state0
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

        state0b = server.goal_start("forall (p: Prop), p -> p")
        del state0b
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

    def test_goal_root(self):
        server = Server()
        state0 = server.goal_start("forall (p: Prop), p -> p")
        e = server.goal_root(state0)
        self.assertEqual(e, None)
        state1 = server.goal_tactic(state0, tactic="exact fun z p => p")
        e = server.goal_root(state1)
        self.assertEqual(e, "fun z p => p")

    def test_automatic_mode(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, tactic="intro a b h")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            "_uniq.17",
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h", t="a ∨ b"),
            ],
            target="b ∨ a",
            name=None,
        )])
        state2 = server.goal_tactic(state1, tactic="cases h")
        self.assertEqual(state2.goals, [
            Goal(
                "_uniq.61",
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            ),
            Goal(
                "_uniq.74",
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="b"),
                ],
                target="b ∨ a",
                name="inr",
            ),
        ])
        state3 = server.goal_tactic(state2, tactic="apply Or.inl", site=Site(goal_id=1))
        state4 = server.goal_tactic(state3, tactic="assumption")
        self.assertEqual(state4.goals, [
            Goal(
                "_uniq.61",
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            )
        ])

    def test_have(self):
        server = Server()
        state0 = server.goal_start("1 + 1 = 2")
        state1 = server.goal_tactic(state0, tactic=TacticHave(branch="2 = 1 + 1", binder_name="h"))
        self.assertEqual(state1.goals, [
            Goal(
                "_uniq.257",
                variables=[],
                target="2 = 1 + 1",
            ),
            Goal(
                "_uniq.259",
                variables=[Variable(name="h", t="2 = 1 + 1")],
                target="1 + 1 = 2",
            ),
        ])
    def test_let(self):
        server = Server()
        state0 = server.goal_start("1 + 1 = 2")
        state1 = server.goal_tactic(
            state0, tactic=TacticLet(branch="2 = 1 + 1", binder_name="h"))
        self.assertEqual(state1.goals, [
            Goal(
                "_uniq.257",
                variables=[],
                name="h",
                target="2 = 1 + 1",
            ),
            Goal(
                "_uniq.259",
                variables=[Variable(name="h", t="2 = 1 + 1", v="?h")],
                target="1 + 1 = 2",
            ),
        ])

    def test_conv_calc(self):
        server = Server(options={"automaticMode": False})
        state0 = server.goal_start("∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b")

        variables = [
            Variable(name="a", t="Nat"),
            Variable(name="b", t="Nat"),
            Variable(name="h", t="b = 2"),
        ]
        state1 = server.goal_tactic(state0, "intro a b h")
        state1b = server.goal_tactic(state1, TacticMode.CALC)
        state2 = server.goal_tactic(state1b, "1 + a + 1 = a + 1 + 1")
        self.assertEqual(state2.goals, [
            Goal(
                "_uniq.381",
                variables,
                target="1 + a + 1 = a + 1 + 1",
                name='calc',
            ),
            Goal(
                "_uniq.400",
                variables,
                target="a + 1 + 1 = a + b",
                mode=TacticMode.CALC,
            ),
        ])
        state_c1 = server.goal_tactic(state2, TacticMode.CONV)
        state_c2 = server.goal_tactic(state_c1, "rhs")
        state_c3 = server.goal_tactic(state_c2, "rw [Nat.add_comm]")
        state_c4 = server.goal_tactic(state_c3, TacticMode.TACTIC)
        #state_c4b = server.goal_resume(state_c4, [state2.goals[0]])
        state_c5 = server.goal_tactic(state_c4, "rfl")
        self.assertTrue(state_c5.is_solved)

        state3 = server.goal_tactic(state2, "_ = a + 2", site=Site(1))
        state4 = server.goal_tactic(state3, "rw [Nat.add_assoc]")
        self.assertTrue(state4.is_solved)

    def test_dependent_mvars(self):
        server = Server(options={"printDependentMVars": True})
        state = server.goal_start("∃ (x : Nat), x + 1 = 0")
        state = server.goal_tactic(state, "apply Exists.intro")
        self.assertEqual(state.goals[0].sibling_dep, {1})
        self.assertEqual(state.goals[1].sibling_dep, set())

    def test_subsume(self):
        server = Server()
        state0 = server.goal_start("forall (p : Prop), p -> p")
        state1 = server.goal_tactic(state0, "intro p")
        state2 = server.goal_tactic(state1, "intro h")
        state3 = server.goal_tactic(state2, "revert h")
        (state, sub) = server.goal_subsume(state3, state3.goals[0], [state1.goals[0], state2.goals[0]])
        self.assertEqual(sub, Subsumption.CYCLE)
        self.assertEqual(state, None)

    def test_env_add_inspect(self):
        server = Server()
        server.env_add(
            name="mystery",
            levels=[],
            t="forall (n: Nat), Nat",
            v="fun (n: Nat) => n + 1",
            is_theorem=False,
        )
        inspect_result = server.env_inspect(name="mystery")
        self.assertEqual(inspect_result['type'], {'pp': 'Nat → Nat'})

    def test_env_parse(self):
        server = Server()
        head, tail = server.env_parse("intro x; apply a", category="tactic")
        self.assertEqual(head, "intro x")
        self.assertEqual(tail, "; apply a")

    def test_goal_state_pickling(self):
        import tempfile
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        with tempfile.TemporaryDirectory() as td:
            path = td + "/goal-state.pickle"
            server.goal_save(state0, path)
            state0b = server.goal_load(path)
            self.assertEqual(state0b.goals, [
                Goal(
                    "_uniq.9",
                    variables=[
                    ],
                    target="∀ (p q : Prop), p ∨ q → q ∨ p",
                )
            ])

    def test_load_header(self):
        server = Server(imports=[])
        server.load_header("import Init\nopen Nat")
        state0 = server.goal_start("forall (n : Nat), n + 1 = n.succ")
        state1 = server.goal_tactic(state0, "intro")
        state2 = server.goal_tactic(state1, "apply add_one")
        self.assertTrue(state2.is_solved)

    def test_check_compile(self):
        server = Server()
        unit, = server.check_compile("example (p: Prop) : p -> p := id")
        self.assertEqual(unit.messages, [])
        unit, = server.check_compile("example (p: Prop) : p -> p := 1")
        self.assertEqual(unit.messages, [Message(
            pos=Position(1, 30),
            pos_end=Position(1, 31),
            data=
            "numerals are data in Lean, but the expected type is "
            "a proposition\n"
            "  p → p : Prop"
        )
        ])
        unit, = server.check_compile("import Lean\nexample (p: Prop) : p -> p := id", read_header=True)
        self.assertEqual(unit.messages, [])

    def test_load_definitions(self):
        server = Server()
        server.load_definitions(
            "def mystery (x : Nat) := x + 123"
        )
        inspect_result = server.env_inspect(name="mystery")
        self.assertEqual(inspect_result['type'], {'pp': 'Nat → Nat'})

    def test_load_sorry(self):
        server = Server()
        unit, = server.load_sorry("theorem mystery (p: Prop) : p → p := sorry", ignore_values=False)
        #self.assertIsNotNone(unit.goal_state, f"{unit.messages}")
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                "_uniq.3",
                [Variable(name="p", t="Prop")],
                target="p → p",
            ),
        ])
        state1 = server.goal_tactic(state0, tactic="intro h")
        state2 = server.goal_tactic(state1, tactic="exact h")
        self.assertTrue(state2.is_solved)

        state1b = server.goal_tactic(state0, tactic=TacticDraft("by\nhave h1 : Or p p := sorry\nsorry"))
        self.assertEqual(state1b.goals, [
            Goal(
                "_uniq.16",
                [Variable(name="p", t="Prop")],
                target="p ∨ p",
            ),
            Goal(
                "_uniq.18",
                [
                    Variable(name="p", t="Prop"),
                    Variable(name="h1", t="p ∨ p"),
                ],
                target="p → p",
            ),
        ])

    def test_distil_search_target(self):
        server = Server()
        unit, = server.load_sorry("theorem mystery (p: Prop) : p → p := sorry")
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                "_uniq.1",
                [],
                target="∀ (p : Prop), p → p",
            ),
        ])
        state1 = server.goal_tactic(state0, tactic="intro p h")
        state2 = server.goal_tactic(state1, tactic="exact h")
        self.assertTrue(state2.is_solved)

    def test_distil_coupled(self):
        server = Server()
        code = """
        def f : Nat -> Nat := sorry
        theorem property (n : Nat) : f n = n := sorry"""
        unit, = server.load_sorry(code, ignore_values=False)
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                "_uniq.7",
                [],
                name='f',
                target="Nat → Nat",
            ),
            Goal(
                "_uniq.10",
                [Variable(name='n', t='Nat')],
                target="?f n = n",
            ),
        ])

    def test_check_track(self):
        server = Server()
        src = "def f : Nat -> Nat := sorry"
        dst = "def f : Nat -> Nat := fun y => y + y"
        self.assertTrue(server.check_track(src, dst).succeeded)

    def test_refactor_search_target(self):
        code = """
        def f : Nat -> Nat := sorry
        theorem property (n : Nat) : f n = n := sorry"""
        target = """
def f_composite : { f : Nat → Nat // ∀ (n : Nat), f n = n } :=
  sorry"""
        server = Server()
        result = server.refactor_search_target(code)
        self.assertEqual(result, target)

if __name__ == '__main__':
    unittest.main()
