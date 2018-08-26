# Software 1.0 Approach

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
        if not n : return None

        if not n % 5 and not n % 3:
            return('FizzBuzz')
        elif not n % 3:
            return('Fizz')
        elif not n % 5: 
            return('Buzz')
        else:
            return str(n)
        
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
        self.assertEqual(FizzBuzz().func(7),'7')

if __name__=='__main__':
    unittest.main()

        