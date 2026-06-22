import torch
import unittest

from nonet.sudokuer import SudokuJudge


VALID = torch.tensor([
    5,3,0,0,7,0,0,0,0,
    6,0,0,1,9,5,0,0,0,
    0,9,8,0,0,0,0,6,0,
    8,0,0,0,6,0,0,0,3,
    4,0,0,8,0,3,0,0,1,
    7,0,0,0,2,0,0,0,6,
    0,6,0,0,0,0,2,8,0,
    0,0,0,4,1,9,0,0,5,
    0,0,0,0,8,0,0,7,9,
])

SOLVED = torch.tensor([
    5,3,4,6,7,8,9,1,2,
    6,7,2,1,9,5,3,4,8,
    1,9,8,3,4,2,5,6,7,
    8,5,9,7,6,1,4,2,3,
    4,2,6,8,5,3,7,9,1,
    7,1,3,9,2,4,8,5,6,
    9,6,1,5,3,7,2,8,4,
    2,8,7,4,1,9,6,3,5,
    3,4,5,2,8,6,1,7,9,
])

INVALID = torch.tensor([
    5,3,0,0,7,0,0,0,0,
    6,0,0,1,9,5,0,0,0,
    0,9,8,0,0,0,0,6,0,
    8,0,0,0,6,0,0,0,3,
    4,0,0,8,0,3,0,0,1,
    7,0,0,0,2,0,0,0,6,
    0,6,0,0,0,0,2,8,0,
    0,0,0,4,1,9,0,0,5,
    0,0,0,0,8,0,0,7,5,  # duplicate 5 in last row
])


class TestSudokuJudge(unittest.TestCase):
    def setUp(self):
        self.judge = SudokuJudge()

    def test_valid_board_is_valid(self):
        self.assertTrue(self.judge.is_valid(VALID).item())

    def test_solved_board_is_valid(self):
        self.assertTrue(self.judge.is_valid(SOLVED).item())

    def test_invalid_board_is_not_valid(self):
        self.assertFalse(self.judge.is_valid(INVALID).item())

    def test_solved_board_is_solved(self):
        self.assertTrue(self.judge.is_solved(SOLVED).item())

    def test_valid_unsolved_board_is_not_solved(self):
        self.assertFalse(self.judge.is_solved(VALID).item())

    def test_batch(self):
        batch = torch.stack([VALID, SOLVED, INVALID])
        valid = self.judge.is_valid(batch)
        solved = self.judge.is_solved(batch)
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertEqual(solved.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
