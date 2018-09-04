"""Software 1.0 Approach"""

import unittest


class FizzBuzz():
    def func(self, n):
        """Return string representation of input number with following exceptions: 
            - if a number is a multiple of 3, return 'Fizz'
            - if a number is a multiple of 5, return 'Buzz'
            - if q number is a multiple of 3 and 5, return 'FizzBuzz'
        :type n: int
        :rtype: str
        """
        # Logic Explanation
        if n % 3 == 0 and n % 5 == 0:
            return 'fizzbuzz'
        elif n % 3 == 0:
            return 'fizz'
        elif n % 5 == 0:
            return 'buzz'
        else:
            return 'other'


class TestFizzBuzz(unittest.TestCase):
    def test_fizz(self):
        self.assertEqual(FizzBuzz().func(3), 'Fizz')

    def test_buzz(self):
        self.assertEqual(FizzBuzz().func(10), 'Buzz')

    def test_fizzbuzz(self):
        self.assertEqual(FizzBuzz().func(30), 'FizzBuzz')

    def test_none(self):
        self.assertEqual(FizzBuzz().func(None), None)

    def test_seven(self):
        self.assertEqual(FizzBuzz().func(7), '7')


if __name__ == '__main__':
    unittest.main()
