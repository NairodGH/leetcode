class Solution(object):
    # https://leetcode.com/problems/contains-duplicate
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # use set()'s unique values property to check for duplicates
        seen = set()
        return any(num in seen or seen.add(num) for num in nums)
    # https://leetcode.com/problems/valid-anagram/
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # since anagrams are same length + same letter frequencies, just check if both strings are equal when ascii-sorted 
        return sorted(s) == sorted(t)
    # https://leetcode.com/problems/two-sum
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # use a value: index mapping of nums to return the index pair if target - num exists and isnt the same index (or None if it doesnt)
        num_indices = {num: i for i, num in enumerate(nums)}
        return next(([i, num_indices[target - num]] for i, num in enumerate(nums) if target - num in num_indices and i != num_indices[target - num]), None)
    # https://leetcode.com/problems/group-anagrams/
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        # put each str into the sorted str key's (see isAnagram) array value then return the dictionnary as a list
        anagrams = {}
        for str in strs:
            sorted_str = ''.join(sorted(str))
            anagrams[sorted_str] = anagrams.get(sorted_str, []) + [str]
        return list(anagrams.values())
    # https://leetcode.com/problems/top-k-frequent-elements/
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # create a num: numCount dict then reverse sort it off its values to a k-long slice
        frequency = {frequency.get(num, 0) + 1 for num in nums}
        return sorted(frequency, key=frequency.get, reverse=True)[:k]
    # https://leetcode.com/problems/product-of-array-except-self/
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # multiply the cross-running product from the left and right of nums into the respective output crossing indices
        n = len(nums)
        output, left_product, right_product = [1] * n, 1, 1
        for i in range(n):
            output[i] *= left_product
            left_product *= nums[i]
            output[~i] *= right_product
            rightProduct *= nums[~i]
        return output
    # https://leetcode.com/problems/valid-sudoku/
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # using sets for each zones to verify, parse board into them to return False if there's any duplicates
        lines, rows, boxes = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    continue
                if board[i][j] in lines[j] or board[i][j] in rows[i] or board[i][j] in boxes[(i // 3) * 3 + j // 3]:
                    return False
                lines[j].add(board[i][j])
                rows[i].add(board[i][j])
                boxes[(i // 3) * 3 + j // 3].add(board[i][j])
        return True
    # https://neetcode.io/problems/string-encode-and-decode
    def encode(self, strs):
        # combine all the strings into one while adding a \x00 separator after each
        return "".join([s + '\x00' for s in strs])
    def decode(self, s):
        # split the string based on the separators (remove the excessive last one)
        return s.split('\x00')[:-1]
    # https://leetcode.com/problems/longest-consecutive-sequence
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # iterate over a nums set and its sequences to find the longest streak (couldn't use sort since O(n log n))
        nums, longest_streak = set(nums), 0
        for num in nums:
            if num - 1 not in nums:
                current_num = num
                while current_num + 1 in nums:
                    current_num += 1
                longest_streak = max(longest_streak, current_num - num + 1)
        return longest_streak
    # https://leetcode.com/problems/valid-palindrome
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # get the string without non-alphanumerical characters and to lowercase then compare it with its reversed version
        formatted_str = ''.join([char for char in s if char.isalnum()]).lower()
        return formatted_str == formatted_str[::-1]