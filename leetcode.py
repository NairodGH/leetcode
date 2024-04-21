from typing import List, Optional
from math import ceil
from collections import deque

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random

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
        result = []
        for i, num in enumerate(nums):
            if i > 0 and num == nums[i-1]: continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                sum = num + nums[left] + nums[right]
                if sum == 0:
                    result.append([num, nums[left], nums[right]])
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
        stack, result = [], [0 for _ in enumerate(temperatures)]
        for i, temperature in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temperature:
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
    # https://leetcode.com/problems/koko-eating-bananas/
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # split the possible k range in half according to the feasability of k untill it converges on the best one
        left, right = 1, max(piles)
        while left != right:
            k = (left + right) // 2
            if sum([ceil(pile / k) for pile in piles]) <= h: right = k
            else: left = k + 1
        return left
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    def findMin(self, nums: List[int]) -> int:
        # converge left and right to narrow the unsorted part of the array where the minimum must be
        left, right = 0, len(nums) - 1
        while nums[right] < nums[left]:
            mid = (left + right) // 2
            if nums[mid] < nums[right]: right = mid
            else: left = mid + 1
        return nums[left]
    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        # find the pivot like findMin (l.272) then binary search the side where target is like search (l.242)
        left, right = 0, len(nums) - 1
        while nums[right] < nums[left]:
            mid = (left + right) // 2
            if nums[mid] < nums[right]: right = mid
            else: left = mid + 1
        if nums[left] <= target <= nums[-1]: right = len(nums) - 1
        else: right, left = left, 0
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target: return mid
            if target < nums[mid]: right = mid - 1
            elif target > nums[mid]: left = mid + 1
        return -1
    # https://leetcode.com/problems/time-based-key-value-store/
    class TimeMap:

        def __init__(self):
            # init empty dict
            self.time_map = {}

        def set(self, key: str, value: str, timestamp: int) -> None:
            # add the (value, timestamp) tuple to time_map's corresponding key (init to empty array if it didnt exist)
            if key not in self.time_map: self.time_map[key] = []
            self.time_map[key].append((value, timestamp))

        def get(self, key: str, timestamp: int) -> str:
            # check if valid key/timestamp then binary search based on "timestamp_prev <= timestamp"
            if key in self.time_map and self.time_map[key][0][1] <= timestamp:
                left, right = 0, len(self.time_map[key]) - 1
                while left < right:
                    mid = (left + right + 1) // 2
                    if timestamp >= self.time_map[key][mid][1]: left = mid
                    else: right = mid - 1
                return self.time_map[key][left][0]
            return ""
    # https://leetcode.com/problems/median-of-two-sorted-arrays/
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # find the points in each list where left numbers are smaller than right numbers then calculate median off of them
        if len(nums1) > len(nums2): nums1, nums2 = nums2, nums1
        len1, len2 = len(nums1), len(nums2)
        left1, right1 = 0, len1
        while left1 <= right1:
            partition1 = (left1 + right1) // 2
            partition2 = ((len1 + len2 + 1) // 2) - partition1
            max_left_1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            min_right_1 = float('inf') if partition1 == len1 else nums1[partition1]
            max_left_2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            min_right_2 = float('inf') if partition2 == len2 else nums2[partition2]
            if max_left_1 <= min_right_2 and max_left_2 <= min_right_1:
                if (len1 + len2) % 2 == 0: return float(max(max_left_1, max_left_2) + min(min_right_1, min_right_2)) / 2
                else: return max(max_left_1, max_left_2)
            elif max_left_1 > min_right_2: right1 = partition1 - 1
            else: left1 = partition1 + 1

    # Sliding Window

    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    def maxProfit(self, prices: List[int]) -> int:
        # record the max profit while searching for the minimum price
        min_price, max_profit = float('inf'), 0
        for price in prices:
            if price < min_price: min_price = price
            elif price - min_price > max_profit: max_profit = price - min_price
        return max_profit
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        # use a map to move the sliding window's start across s while recording the max length
        char_map = {}
        max_len = start = 0
        for i, char in enumerate(s):
            if char in char_map and start <= char_map[char]: start = char_map[char] + 1
            else: max_len = max(max_len, i - start + 1)
            char_map[char] = i
        return max_len
    # https://leetcode.com/problems/longest-repeating-character-replacement/
    def characterReplacement(self, s: str, k: int) -> int:
        # use a freq map to move a valid (within max freq + k) sliding window across s while recording its max length
        left = right = max_freq = result = 0
        letters_freq = {}
        while right < len(s):
            letters_freq[s[right]] = letters_freq.get(s[right], 0) + 1
            max_freq = max(max_freq, letters_freq[s[right]])
            if right - left + 1 > max_freq + k:
                letters_freq[s[left]] -= 1
                left += 1
            else: result = max(result, right - left + 1)
            right += 1
        return result
    # https://leetcode.com/problems/permutation-in-string/
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # move a s1-sized window over s2 while comparing letter frequencies until they match (True) or not (False)
        s1_freq = {}
        window_freq = {}
        for char in s1: s1_freq[char] = s1_freq.get(char, 0) + 1
        for i, char in enumerate(s2):
            window_freq[char] = window_freq.get(char, 0) + 1
            if i >= len(s1):
                if window_freq[s2[i - len(s1)]] == 1: del window_freq[s2[i - len(s1)]]
                else: window_freq[s2[i - len(s1)]] -= 1
            if window_freq == s1_freq: return True
        return False
    # https://leetcode.com/problems/minimum-window-substring/
    def minWindow(self, s: str, t: str) -> str:
        # move a left/right pointers window accross s while saving (min_start min_len) the smallest valid (t_counter) substring
        if not s or not t or len(s) < len(t): return ""
        letters_freq = [0] * 128
        t_counter = len(t)
        left = right = min_start = 0
        min_len = float('inf')
        for char in t: letters_freq[ord(char)] += 1
        while right < len(s):
            if letters_freq[ord(s[right])] > 0: t_counter -= 1
            letters_freq[ord(s[right])] -= 1
            right += 1
            while t_counter == 0:
                if right - left < min_len:
                    min_start = left
                    min_len = right - left
                if letters_freq[ord(s[left])] == 0: t_counter += 1
                letters_freq[ord(s[left])] += 1
                left += 1
        return "" if min_len == float('inf') else s[min_start:min_start + min_len]
    # https://leetcode.com/problems/sliding-window-maximum/
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # use a dequeue window keeping track of max num indices to store their num in result
        result = []
        window = deque()
        for i, num in enumerate(nums):
            while window and num > nums[window[-1]]: window.pop()
            window.append(i)
            if i + 1 >= k:
                result.append(nums[window[0]])
                if i + 1 - k >= window[0]: window.popleft()
        return result
    
    # Linked List

    # https://leetcode.com/problems/reverse-linked-list/
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # change the next pointer to the prev one for each node
        prev = None
        curr = head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev
    # https://leetcode.com/problems/merge-two-sorted-lists/
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # traverse list1 and list2 while linking their numbers in order to a node, link the rest since necessarily greater
        prehead = curr = ListNode()
        while list1 and list2:
            if list1.val <= list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        curr.next = list1 if list1 is not None else list2
        return prehead.next
    # https://leetcode.com/problems/reorder-list/
    def reorderList(self, head: Optional[ListNode]) -> None:
        # find the middle using slow-fast pointers, reverse the second half and merge both halves alternately
        if not head or not head.next: return head
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        curr = slow
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        left = head
        right = prev
        while right.next:
            temp1 = left.next
            temp2 = right.next
            left.next = right
            right.next = temp1
            left = temp1
            right = temp2
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # advance pointer A n times then A and B until A reaches the end so that B is nth from the end, remove node at B
        first = second = dummy = ListNode(next=head)
        for _ in range(n + 1): first = first.next
        while first is not None:
            second = second.next
            first = first.next
        second.next = second.next.next if second.next is not None else None
        return dummy.next
    # https://leetcode.com/problems/copy-list-with-random-pointer/
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        # create nodes copies without random pointers, set their next/random pointers and map them to the original ones
        if not head: return None
        node_map = {}
        current = head
        while current:
            node_map[current] = Node(current.val)
            current = current.next
        current = head
        while current:
            if current.next: node_map[current].next = node_map[current.next]
            if current.random: node_map[current].random = node_map[current.random]
            current = current.next
        return node_map[head]
    # https://leetcode.com/problems/add-two-numbers/
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # go through l1 and l2 and create the sum linked list while keeping track of the carry
        prehead = current = ListNode()
        carry = 0
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            current.next = ListNode(digit)
            current = current.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return prehead.next