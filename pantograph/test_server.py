from .server import *
import unittest

class TestServer(unittest.TestCase):

    def test_version(self):
        """
        NOTE: Update this after upstream updates.
        """
        self.assertEqual(get_version(), "0.3.9")

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
                "_uniq.256",
                variables=[],
                target="2 = 1 + 1",
            ),
            Goal(
                "_uniq.258",
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
                "_uniq.256",
                variables=[],
                name="h",
                target="2 = 1 + 1",
            ),
            Goal(
                "_uniq.258",
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
                "_uniq.372",
                variables,
                target="1 + a + 1 = a + 1 + 1",
                name='calc',
            ),
            Goal(
                "_uniq.391",
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
        src = state1.goals[0]
        (sub, state, subsumptor) = server.goal_subsume(
            state3,
            state3.goals[0],
            [state1.goals[0], state2.goals[0]],
        )
        self.assertEqual(sub, Subsumption.CYCLE)
        self.assertEqual(state, None)
        self.assertEqual(subsumptor, src)

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
        unit, = server.load_sorry("theorem mystery (p: Prop) : p → p := sorry")
        #self.assertIsNotNone(unit.goal_state, f"{unit.messages}")
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                "_uniq.5",
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
                "_uniq.19",
                [Variable(name="p", t="Prop")],
                target="p ∨ p",
            ),
            Goal(
                "_uniq.21",
                [
                    Variable(name="p", t="Prop"),
                    Variable(name="h1", t="p ∨ p", v="?m.7"),
                ],
                target="p → p",
            ),
        ])

    def test_distil_search_target(self):
        server = Server()
        unit, = server.load_sorry("theorem mystery (p: Prop) : p → p := sorry", ignore_values = True)
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                "_uniq.3",
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
