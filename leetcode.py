class Solution(object):
    # https://leetcode.com/problems/contains-duplicate/description/
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
    # https://leetcode.com/problems/two-sum/description/
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
            sortedStr = ''.join(sorted(str))
            anagrams[sortedStr] = anagrams.get(sortedStr, []) + [str]
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