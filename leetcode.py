from typing import List

class Solution(object):

    # https://neetcode.io/roadmap

    # Arrays & Hashing

    # https://leetcode.com/problems/contains-duplicate
    def containsDuplicate(self, nums: List[int]) -> bool:
        # use set()'s unique values property to check for duplicates by comparing the length
        return len(nums) > len(set(nums))
    # https://leetcode.com/problems/valid-anagram/
    def isAnagram(self, s: str, t: str) -> bool:
        # check if both strings have the same length and letter frequency with a set
        if len(s) != len(t): return False
        for char in set(s):
            if s.count(char) != t.count(char): return False
        return True
    # https://leetcode.com/problems/two-sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # use a value: index mapping of nums to return the index pair if target - num exists and isnt the same index (or None if it doesnt)
        num_indices = {num: i for i, num in enumerate(nums)}
        return next(([i, num_indices[target - num]] for i, num in enumerate(nums) if target - num in num_indices and i != num_indices[target - num]), None)
    # https://leetcode.com/problems/group-anagrams/
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # put each str into the sorted str key's (see isAnagram) array value then return the dictionnary as a list
        anagrams = {}
        for str in strs:
            sorted_str = ''.join(sorted(str))
            if sorted_str in anagrams: anagrams[sorted_str].append(str)
            else: anagrams[sorted_str] = [str]
        return list(anagrams.values())
    # https://leetcode.com/problems/top-k-frequent-elements/
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # create a num: numCount dict then reverse sort it off its values to a k-long slice
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        return sorted(frequency, key=frequency.get, reverse=True)[:k]
    # https://leetcode.com/problems/product-of-array-except-self/
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # multiply the cross-running product from the left and right of nums into the respective output crossing indices
        n = len(nums)
        output, left_product, right_product = [1] * n, 1, 1
        for i in range(n):
            output[i] *= left_product
            left_product *= nums[i]
            output[~i] *= right_product
            right_product *= nums[~i]
        return output
    # https://leetcode.com/problems/valid-sudoku/
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # using lists for each zones to verify, parse board into them to return False if there's any duplicates
        rows, lines, boxes = [[] for _ in range(9)], [[] for _ in range(9)], [[] for _ in range(9)]
        for i in range(9):
            k = (i // 3) * 3
            for j in range(9):
                if board[i][j] == '.': continue
                if board[i][j] in rows[i] or board[i][j] in lines[j]: return False
                curr_k = k + (j // 3)
                if board[i][j] in boxes[curr_k]: return False
                rows[i].append(board[i][j])
                lines[j].append(board[i][j])
                boxes[curr_k].append(board[i][j])
        return True
    # https://neetcode.io/problems/string-encode-and-decode
    def encode(self, strs: List[str]) -> str:
        # combine all the strings into one while adding a \x00 separator after each
        return "".join([s + '\x00' for s in strs])
    def decode(self, s: str) -> List[str]:
        # split the string based on the separators (remove the excessive last one)
        return s.split('\x00')[:-1]
    # https://leetcode.com/problems/longest-consecutive-sequence
    def longestConsecutive(self, nums: List[int]) -> int:
        # iterate over a nums set and its sequences to find the longest streak (couldn't use sort since O(n log n))
        nums, longest_streak = set(nums), 0
        for num in nums:
            if num - 1 not in nums:
                current_num = num
                while current_num + 1 in nums: current_num += 1
                longest_streak = max(longest_streak, current_num - num + 1)
        return longest_streak
    # https://leetcode.com/problems/valid-palindrome

    # Two Pointers

    def isPalindrome(self, s: str) -> bool:
        # get the alphanumerical + lowercase string then compare each character with its mirrored equivalent
        formatted_str = ''.join([char for char in s if char.isalnum()]).lower()
        return all([formatted_str[i] == formatted_str[-(i + 1)] for i in range(len(formatted_str) // 2)])
    # https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # create 2 pointers at the start/end and move each toward the other based on the sum result until it's found
        left, right = 0, len(numbers) - 1
        while left < right:
            sum = numbers[left] + numbers[right]
            if sum == target: return [left + 1, right + 1]
            elif sum < target: left += 1
            else: right -= 1
    # https://leetcode.com/problems/3sum/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # using a sort (for 2-pointer), replicate twoSum but with an extra fixed value and anti-duplicate checks
        nums.sort()
        size, result = len(nums), []
        for i in range(size):
            if i > 0 and nums[i] == nums[i-1]: continue
            left, right = i + 1, size - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]: left += 1
                    while left < right and nums[right] == nums[right-1]: right -= 1
                    left, right = left + 1, right - 1
                elif sum < 0: left += 1
                else: right -= 1
        return result
    # https://leetcode.com/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        # get iteratively the best area by moving the lowest height pointer and stopping when there's no possible best area
        left, right, best_area, max_height = 0, len(height) - 1, 0, max(height)
        while left < right:
            best_area = max(min(height[left], height[right]) * (right - left), best_area)
            if height[left] < height[right]: left += 1
            else: right -= 1
            if best_area > max_height * (right - left): break
        return best_area
    # https://leetcode.com/problems/trapping-rain-water/
    def trap(self, height: List[int]) -> int:
        # at each step, converge left and right while adding their max and current height substraction in the total
        left, right, total = 0, len(height) - 1, 0
        left_max, right_max = height[left], height[right]
        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                total += (left_max - height[left])
            else:
                right -= 1
                right_max = max(right_max, height[right])
                total += (right_max - height[right])
        return total

    # Stack

    # https://leetcode.com/problems/valid-parentheses/
    def isValid(self, s: str) -> bool:
        # push opening brackets and pop closing ones if they match, the stack should then be empty at the end
        stack, correspondance = [], {'(': ')', '[': ']', '{': '}'}
        for char in s:
            if char in correspondance: stack.append(char)
            elif not len(stack) or char != correspondance[stack.pop()]: return False
        return not len(stack)
    # https://leetcode.com/problems/min-stack/
    class MinStack:

        def __init__(self):
            # init empty stack (main) and minimums stack (used for getMin)
            self.stack = []
            self.min_stack = []
        
        def push(self, val: int) -> None:
            # push val to stack, and to minimums stack if it's empty or a new minimum
            self.stack.append(val)
            if not self.min_stack or val <= self.min_stack[-1]: self.min_stack.append(val)
        
        def pop(self) -> None:
            # pop val off stack, and off minimums stack if it's the minimum
            result = self.stack.pop()
            if result == self.min_stack[-1]: self.min_stack.pop()
        
        def top(self) -> int:
            # peek the last stack value
            return self.stack[-1]
        
        def getMin(self) -> int:
            # peek the stack minimum
            return self.min_stack[-1]
    # https://leetcode.com/problems/evaluate-reverse-polish-notation/
    def evalRPN(self, tokens: List[str]) -> int:
        # for each token, push the number or execute the operation on the last 2 numbers (extra steps for RPN's divisions)
        stack = []
        for token in tokens:
            if token in ["+", "-", "*", "/"]:
                num2, num1 = stack.pop(), stack.pop()
                if token == "+": stack.append(num1 + num2)
                elif token == "-": stack.append(num1 - num2)
                elif token == "*": stack.append(num1 * num2)
                elif num1 * num2 >= 0: stack.append(num1 // num2)
                else: stack.append(-(-num1 // num2))
            else: stack.append(int(token))
        return stack[0]
    # https://leetcode.com/problems/generate-parentheses/
    def generateParenthesis(self, n: int) -> List[str]:
        # use a tracking tuples stack to add the right parentheses and complete combinations
        stack, result = [("(", 1, 0)], []
        while stack:
            s, open, close = stack.pop()
            if open == close == n: result.append(s)
            else:
                if open < n: stack.append((s + "(", open + 1, close))
                if close < open: stack.append((s + ")", open, close + 1))
        return result
    # https://leetcode.com/problems/daily-temperatures/
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # use an indices stack of temperatures waiting for a warmer day, put the index differences in result when there is
        stack, result = [], [0 for _ in range(len(temperatures))]
        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                index = stack.pop()
                result[index] = i - index
            stack.append(i)
        return result
    # https://leetcode.com/problems/car-fleet/
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        # sort cars by position and store in a stack each different arrival times (fleets) based on if they can catch up
        stack = []
        for pos, spd in sorted(zip(position, speed), reverse=True):
            time = (target - pos) / spd
            if not stack: stack.append(time)
            elif time > stack[-1]: stack.append(time)
        return len(stack)
    # https://leetcode.com/problems/largest-rectangle-in-histogram/
    def largestRectangleArea(self, heights: List[int]) -> int:
        # stack (position, height) infos and store max area for each lower height, then check for the rest just in case
        stack, max_area = [], 0
        for index, height in enumerate(heights):
            start = index
            while stack and stack[-1][1] > height:
                popped_index, popped_height = stack.pop()
                max_area = max(max_area, popped_height * (index - popped_index))
                start = popped_index
            stack.append((start, height))
        for index, height in stack: max_area = max(max_area, height * (len(heights) - index))
        return max_area
    
    # Binary Search

    # https://leetcode.com/problems/binary-search/
    def search(self, nums: List[int], target: int) -> int:
        # split the search range in half according to the target until its found or not (-1)
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target: return mid
            if target < nums[mid]: right = mid - 1
            elif target > nums[mid]: left = mid + 1
        return -1
    # https://leetcode.com/problems/search-a-2d-matrix/
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # same as search above but bring matrix down to 1D and return True/False instead of target index/-1
        matrix = [item for sublist in matrix for item in sublist]
        left, right = 0, len(matrix) - 1
        while left <= right:
            mid = (left + right) // 2
            if matrix[mid] == target: return True
            if target < matrix[mid]: right = mid - 1
            elif target > matrix[mid]: left = mid + 1
        return False