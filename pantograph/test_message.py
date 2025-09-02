from .message import *
import unittest

class TestMessage(unittest.TestCase):

    def test_severity(self):
        self.assertEqual(str(Severity.ERROR), "error")
    def test_message(self):
        m = Message(
            data="hi",
            pos=Position(12, 34),
            pos_end=Position(56, 78),
            severity=Severity.WARNING,
        )
        self.assertEqual(str(m), "12:34-56:78: warning: hi")

        m = Message(
            data="hi",
            pos=Position(0, 0),
            severity=Severity.ERROR,
        )
        self.assertEqual(str(m), "0:0: error: hi")

if __name__ == '__main__':
    unittest.main()
