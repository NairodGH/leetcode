from typing import List, Optional
from math import ceil, comb
from collections import deque, OrderedDict, Counter, defaultdict
from heapq import heapify, heappop, heappush, heapreplace
from bisect import bisect_left

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

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
        if len(s) != len(t):
            return False
        for char in set(s):
            if s.count(char) != t.count(char):
                return False
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
            if sorted_str in anagrams:
                anagrams[sorted_str].append(str)
            else:
                anagrams[sorted_str] = [str]
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
                if board[i][j] == '.':
                    continue
                if board[i][j] in rows[i] or board[i][j] in lines[j]:
                    return False
                curr_k = k + (j // 3)
                if board[i][j] in boxes[curr_k]:
                    return False
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
                while current_num + 1 in nums:
                    current_num += 1
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
            if sum == target:
                return [left + 1, right + 1]
            elif sum < target:
                left += 1
            else:
                right -= 1
    # https://leetcode.com/problems/3sum/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # using a sort (for 2-pointer), replicate twoSum but with an extra fixed value and anti-duplicate checks
        nums.sort()
        result = []
        for i, num in enumerate(nums):
            if i > 0 and num == nums[i-1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                sum = num + nums[left] + nums[right]
                if sum == 0:
                    result.append([num, nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left, right = left + 1, right - 1
                elif sum < 0:
                    left += 1
                else:
                    right -= 1
        return result
    # https://leetcode.com/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        # get iteratively the best area by moving the lowest height pointer and stopping when there's no possible best area
        left, right, best_area, max_height = 0, len(height) - 1, 0, max(height)
        while left < right:
            best_area = max(min(height[left], height[right]) * (right - left), best_area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            if best_area > max_height * (right - left):
                break
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
            if char in correspondance:
                stack.append(char)
            elif not len(stack) or char != correspondance[stack.pop()]:
                return False
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
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)
        
        def pop(self) -> None:
            # pop val off stack, and off minimums stack if it's the minimum
            result = self.stack.pop()
            if result == self.min_stack[-1]:
                self.min_stack.pop()
        
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
                if token == "+":
                    stack.append(num1 + num2)
                elif token == "-":
                    stack.append(num1 - num2)
                elif token == "*":
                    stack.append(num1 * num2)
                elif num1 * num2 >= 0:
                    stack.append(num1 // num2)
                else:
                    stack.append(-(-num1 // num2))
            else:
                stack.append(int(token))
        return stack[0]
    # https://leetcode.com/problems/generate-parentheses/
    def generateParenthesis(self, n: int) -> List[str]:
        # use a tracking tuples stack to add the right parentheses and complete combinations
        stack, result = [("(", 1, 0)], []
        while stack:
            s, open, close = stack.pop()
            if open == close == n:
                result.append(s)
            else:
                if open < n:
                    stack.append((s + "(", open + 1, close))
                if close < open:
                    stack.append((s + ")", open, close + 1))
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
            if not stack:
                stack.append(time)
            elif time > stack[-1]:
                stack.append(time)
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
        for index, height in stack:
            max_area = max(max_area, height * (len(heights) - index))
        return max_area
    
    # Binary Search

    # https://leetcode.com/problems/binary-search/
    def search(self, nums: List[int], target: int) -> int:
        # split the search range in half according to the target until its found or not (-1)
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if target < nums[mid]:
                right = mid - 1
            elif target > nums[mid]:
                left = mid + 1
        return -1
    # https://leetcode.com/problems/search-a-2d-matrix/
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # same as search above but bring matrix down to 1D and return True/False instead of target index/-1
        matrix = [item for sublist in matrix for item in sublist]
        left, right = 0, len(matrix) - 1
        while left <= right:
            mid = (left + right) // 2
            if matrix[mid] == target:
                return True
            if target < matrix[mid]:
                right = mid - 1
            elif target > matrix[mid]:
                left = mid + 1
        return False
    # https://leetcode.com/problems/koko-eating-bananas/
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # split the possible k range in half according to the feasability of k untill it converges on the best one
        left, right = 1, max(piles)
        while left != right:
            k = (left + right) // 2
            if sum([ceil(pile / k) for pile in piles]) <= h:
                right = k
            else:
                left = k + 1
        return left
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    def findMin(self, nums: List[int]) -> int:
        # converge left and right to narrow the unsorted part of the array where the minimum must be
        left, right = 0, len(nums) - 1
        while nums[right] < nums[left]:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]
    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        # find the pivot like findMin (l.272) then binary search the side where target is like search (l.242)
        left, right = 0, len(nums) - 1
        while nums[right] < nums[left]:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        if nums[left] <= target <= nums[-1]:
            right = len(nums) - 1
        else:
            right, left = left, 0
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if target < nums[mid]:
                right = mid - 1
            elif target > nums[mid]:
                left = mid + 1
        return -1
    # https://leetcode.com/problems/time-based-key-value-store/
    class TimeMap:

        def __init__(self):
            # init empty dict
            self.time_map = {}

        def set(self, key: str, value: str, timestamp: int) -> None:
            # add the (value, timestamp) tuple to time_map's corresponding key (init to empty array if it didnt exist)
            if key not in self.time_map:
                self.time_map[key] = []
            self.time_map[key].append((value, timestamp))

        def get(self, key: str, timestamp: int) -> str:
            # check if valid key/timestamp then binary search based on "timestamp_prev <= timestamp"
            if key in self.time_map and self.time_map[key][0][1] <= timestamp:
                left, right = 0, len(self.time_map[key]) - 1
                while left < right:
                    mid = (left + right + 1) // 2
                    if timestamp >= self.time_map[key][mid][1]:
                        left = mid
                    else:
                        right = mid - 1
                return self.time_map[key][left][0]
            return ""
    # https://leetcode.com/problems/median-of-two-sorted-arrays/
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # find the points in each list where left numbers are smaller than right numbers then calculate median off of them
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
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
                if (len1 + len2) % 2 == 0:
                    return float(max(max_left_1, max_left_2) + min(min_right_1, min_right_2)) / 2
                else:
                    return max(max_left_1, max_left_2)
            elif max_left_1 > min_right_2:
                right1 = partition1 - 1
            else:
                left1 = partition1 + 1

    # Sliding Window

    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    def maxProfit(self, prices: List[int]) -> int:
        # record the max profit while searching for the minimum price
        min_price, max_profit = float('inf'), 0
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_profit:
                max_profit = price - min_price
        return max_profit
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        # use a map to move the sliding window's start across s while recording the max length
        char_map = {}
        max_len = start = 0
        for i, char in enumerate(s):
            if char in char_map and start <= char_map[char]:
                start = char_map[char] + 1
            else:
                max_len = max(max_len, i - start + 1)
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
            else:
                result = max(result, right - left + 1)
            right += 1
        return result
    # https://leetcode.com/problems/permutation-in-string/
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # move a s1-sized window over s2 while comparing letter frequencies until they match (True) or not (False)
        s1_freq = {}
        window_freq = {}
        for char in s1:
            s1_freq[char] = s1_freq.get(char, 0) + 1
        for i, char in enumerate(s2):
            window_freq[char] = window_freq.get(char, 0) + 1
            if i >= len(s1):
                if window_freq[s2[i - len(s1)]] == 1:
                    del window_freq[s2[i - len(s1)]]
                else:
                    window_freq[s2[i - len(s1)]] -= 1
            if window_freq == s1_freq:
                return True
        return False
    # https://leetcode.com/problems/minimum-window-substring/
    def minWindow(self, s: str, t: str) -> str:
        # move a left/right pointers window accross s while saving (min_start min_len) the smallest valid (t_counter) substring
        if not s or not t or len(s) < len(t):
            return ""
        letters_freq = [0] * 128
        t_counter = len(t)
        left = right = min_start = 0
        min_len = float('inf')
        for char in t:
            letters_freq[ord(char)] += 1
        while right < len(s):
            if letters_freq[ord(s[right])] > 0:
                t_counter -= 1
            letters_freq[ord(s[right])] -= 1
            right += 1
            while t_counter == 0:
                if right - left < min_len:
                    min_start = left
                    min_len = right - left
                if letters_freq[ord(s[left])] == 0:
                    t_counter += 1
                letters_freq[ord(s[left])] += 1
                left += 1
        return "" if min_len == float('inf') else s[min_start:min_start + min_len]
    # https://leetcode.com/problems/sliding-window-maximum/
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # use a dequeue window keeping track of max num indices to store their num in result
        result = []
        window = deque()
        for i, num in enumerate(nums):
            while window and num > nums[window[-1]]:
                window.pop()
            window.append(i)
            if i + 1 >= k:
                result.append(nums[window[0]])
                if i + 1 - k >= window[0]:
                    window.popleft()
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
        if not head or not head.next:
            return head
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
        for _ in range(n + 1):
            first = first.next
        while first is not None:
            second = second.next
            first = first.next
        second.next = second.next.next if second.next is not None else None
        return dummy.next
    # https://leetcode.com/problems/copy-list-with-random-pointer/
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        # create nodes copies without random pointers, set their next/random pointers and map them to the original ones
        if not head:
            return None
        node_map = {}
        current = head
        while current:
            node_map[current] = Node(current.val)
            current = current.next
        current = head
        while current:
            if current.next:
                node_map[current].next = node_map[current.next]
            if current.random:
                node_map[current].random = node_map[current.random]
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
    # https://leetcode.com/problems/linked-list-cycle/
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # move slow and fast pointers until fast reaches the end, if fast is ever back to slow then there's a cycle
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while fast and fast.next:
            if slow == fast:
                return True
            slow = slow.next
            fast = fast.next.next
        return False
    # https://leetcode.com/problems/find-the-duplicate-number/
    def findDuplicate(self, nums: List[int]) -> int:
        # treat numbers as pointers and the List as linked, use slow and fast pointer to find the cycle (duplicate) like in hasCycle
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow
    # https://leetcode.com/problems/lru-cache/
    class LRUCache:
        # initialize the OrderedDict and set the capacity
        def __init__(self, capacity: int):
            self.cache = OrderedDict()
            self.capacity = capacity

        # return the cache's key value if it exist while moving it to the end (most recent used key)
        def get(self, key: int) -> int:
            if key not in self.cache:
                return -1
            self.cache.move_to_end(key)
            return self.cache[key]

        # change the cache's key value while moving it to then end, if the capacity is exceeded pop the first (LRU) key
        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(False)
    # https://leetcode.com/problems/merge-k-sorted-lists/
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # define a function to merge 2 lists and divide and conquer through lists with it by pushing results into mergedLists until it's 1D
        def mergeTwoLists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode(0)
            curr = dummy
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = l1
                    l1 = l1.next
                else:
                    curr.next = l2
                    l2 = l2.next
                curr = curr.next
            curr.next = l1 if l1 else l2
            return dummy.next

        while len(lists) > 1:
            mergedLists = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                mergedLists.append(mergeTwoLists(l1, l2))
            lists = mergedLists
        return lists[0] if lists else None
    # https://leetcode.com/problems/reverse-nodes-in-k-group/
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # go through each k-sized groups of the list (using left-right ranges and jump sizes) while reversing them until there isn't any group left
        dummy = jump = ListNode(0)
        dummy.next = left = right = head
        while True:
            count = 0
            while right and count < k:
                right = right.next
                count += 1
            if count == k:
                prev, curr = right, left
                for _ in range(k):
                    curr.next, curr, prev = prev, curr.next, curr
                jump.next, jump, left = prev, left, right
            else:
                return dummy.next
    
    # Trees

    # https://leetcode.com/problems/invert-binary-tree/
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # recursively swap left and right pointers until we reach the end of the tree
        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
    # https://leetcode.com/problems/maximum-depth-of-binary-tree/
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # use a deque to store root+depth tuples along the tree while increasing depths and storing the max
        if not root:
            return 0
        queue = deque([(root, 1)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
        return max_depth
    # https://leetcode.com/problems/diameter-of-binary-tree/
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # DFS through the tree while saving the max diameter from the recursive left+right diameters
        self.diameter = 0
        def DFS(node):
            if not node:
                return 0
            left_diameter = DFS(node.left)
            right_diameter = DFS(node.right)
            self.diameter = max(self.diameter, left_diameter + right_diameter)
            return 1 + max(left_diameter, right_diameter)
        DFS(root)
        return self.diameter
    # https://leetcode.com/problems/balanced-binary-tree/
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # DFS through the tree by counting each level from the recursive max(left, right) + 1 until the difference is more than 1 (unbalanced) or not
        def DFS(node):
            if not node:
                return 0
            left, right = DFS(node.left), DFS(node.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return max(left, right) + 1
        return DFS(root) != -1
    # https://leetcode.com/problems/same-tree/
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # recursively check left and right of both trees at the same time, stoping at any equality until we're sure both are the same
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    # https://neetcode.io/problems/subtree-of-a-binary-tree
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # recursively check if left or right are equal to the subRoot using isSameTree
        if not root:
            return False
        if self.isSameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # use the BST property to find the LCA and return it
        current = root
        while current:
            if p.val < current.val and q.val < current.val:
                current = current.left
            elif p.val > current.val and q.val > current.val:
                current = current.right
            else:
                return current
    # https://leetcode.com/problems/binary-tree-level-order-traversal/
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # use a recursive support function to traverse the tree from root while recording values at their levels
        if not root:
            return []
        result = []
        def traverse(node: TreeNode, level: int):
            if len(result) == level:
                result.append([])
            result[level].append(node.val)
            if node.left:
                traverse(node.left, level + 1)
            if node.right:
                traverse(node.right, level + 1)
        traverse(root, 0)
        return result
    # https://leetcode.com/problems/binary-tree-right-side-view/
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # DFS through the tree while getting values at their levels (right first for right side, leetcode trees design)
        result = []
        def DFS(node, level):
            if not node:
                return
            if level == len(result):
                result.append(node.val)
            DFS(node.right, level + 1)
            DFS(node.left, level + 1)
        DFS(root, 0)
        return result
    # https://leetcode.com/problems/count-good-nodes-in-binary-tree/
    def goodNodes(self, root: TreeNode) -> int:
        # DFS through the tree while adding to count (array alloc trick) whenever val is greater than the prev's (good node)
        if not root:
            return 0
        count = [0]
        def DFS(node, curMax):
            if not node:
                return
            if node.val >= curMax:
                count[0] += 1
                curMax = node.val
            DFS(node.left, curMax)
            DFS(node.right, curMax)
        DFS(root, root.val)
        return count[0]
    # https://leetcode.com/problems/validate-binary-search-tree/
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # recursively validate the tree's BST properties with a support function
        def validate(node, low=float('-inf'), high=float('inf')):
            if not node:
                return True
            if not low < node.val < high:
                return False
            return validate(node.left, low, node.val) and validate(node.right, node.val, high)
        return validate(root)
    # https://leetcode.com/problems/kth-smallest-element-in-a-bst/
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # recursive in-order (left, root, right) traversal through the tree, return when k-th found (smallest since BST property)
        self.k = k
        self.result = None
        def sort(node):
            if node is None:
                return
            sort(node.left)
            self.k -= 1
            if self.k == 0:
                self.result = node.val
                return
            sort(node.right)
        sort(root)
        return self.result
    # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # use an id hashmap of inorder to build the tree with a recursive helper func that determines the result off of in-order/pre-order traversal properties
        in_ids = {val: id for id, val in enumerate(inorder)}
        def helper(pre_left, pre_right, in_left, in_right):
            if pre_left > pre_right:
                return None
            root_val = preorder[pre_left]
            root = TreeNode(root_val)
            in_root_id = in_ids[root_val]
            left_size = in_root_id - in_left
            root.left = helper(pre_left + 1, pre_left + left_size, in_left, in_root_id - 1)
            root.right = helper(pre_left + left_size + 1, pre_right, in_root_id + 1, in_right)
            return root
        return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
    # https://leetcode.com/problems/binary-tree-maximum-path-sum/
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # DFS through the tree while recording the max_sum along pathes
        max_sum = float('-inf')
        def DFS(node):
            nonlocal max_sum
            if not node:
                return 0
            left_gain = max(DFS(node.left), 0)
            right_gain = max(DFS(node.right), 0)
            price_newpath = node.val + left_gain + right_gain
            max_sum = max(max_sum, price_newpath)
            return node.val + max(left_gain, right_gain)
        DFS(root)
        return max_sum
    # https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
    class Codec:
        # DFS through root in pre-order to build the string representation of the tree with its comma-separated values
        def serialize(self, root):
            def DFS(node):
                if not node:
                    return 'null,'
                return str(node.val) + ',' + DFS(node.left) + DFS(node.right)
            return DFS(root)[:-1]
        
        # DFS through the comma-separated string (using iter) to reconstruct the tree
        def deserialize(self, data):
            def DFS(nodes):
                val = next(nodes)
                if val == 'null':
                    return None
                node = TreeNode(int(val))
                node.left = DFS(nodes)
                node.right = DFS(nodes)
                return node
            return DFS(iter(data.split(',')))

    # Tries

    # https://leetcode.com/problems/implement-trie-prefix-tree/
    class Trie:
        # create the root of the Trie being a dictionnary that'll hold the different nested characters pathes that represent the words ending with a delimiter
        def __init__(self):
            self.root = {}

        # for each character in the word, add a dictionnary if it doesnt already exist and nest in it until reaching the end to add the delimiter
        def insert(self, word: str) -> None:
            node = self.root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['\0'] = True

        # search each character of the word, if any doesnt exist return false, if we went through all and found the delimiter then return true
        def search(self, word: str) -> bool:
            node = self.root
            for char in word:
                if char not in node:
                    return False
                node = node[char]
            return '\0' in node

        # same as search but returns true instantly instead of checking for the delimiter (ex: search(a) with a->b->\0 is false but startsWith would be true)
        def startsWith(self, prefix: str) -> bool:
            node = self.root
            for char in prefix:
                if char not in node:
                    return False
                node = node[char]
            return True
    # https://leetcode.com/problems/design-add-and-search-words-data-structure/
    class WordDictionary:
        # same as Trie class
        def __init__(self):
            self.root = {}

        # same as Trie class
        def addWord(self, word: str) -> None:
            node = self.root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['\0'] = True

        # '.' wildcards make it different from Trie class' search as we need to DFS through each possibility on the remainder of the word but the rest is the same
        def search(self, word: str) -> bool:
            def DFS(word, node):
                for i, char in enumerate(word):
                    if char == '.':
                        return any(DFS(word[i+1:], node[child]) for child in node if child != '\0')
                    if char not in node:
                        return False
                    node = node[char]
                return '\0' in node
            return DFS(word, self.root)
    # https://leetcode.com/problems/word-search-ii/
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # build a Trie out of words then DFS through each in the board by going in all 4 directions, mark visited cells and once found add words to the set
        trie = self.Trie()
        for word in words:
            trie.insert(word)
        result = set()
        def DFS(node, i, j, path):
            char = board[i][j]
            node = node[char]
            if '\0' in node:
                result.add(path + char)
                del node['\0'] # avoid further exploration since the word is already found
            board[i][j] = '#' # mark as visited, no word can have '#' so it'll just stop the dfs
            for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)]: # go in all 4 directions
                ni, nj = i + x, j + y
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] in node:
                    DFS(node, ni, nj, path + char) # if within bounds & valid char, DFS in that direction
            board[i][j] = char  # unmark as visited
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] in trie.root: # only start DFS from possible word pathes
                    DFS(trie.root, i, j, "")
        return list(result)

    # Backtracking

    # https://leetcode.com/problems/subsets/
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # sort then backtrack by adding each of the nums to each of the pathes' copies (0:[] 1:[],[1] 2:[],[2],[1],[1, 2] 3:...)
        def backtrack(start, path):
            res.append(path)
            for i in range(start, len(nums)):
                backtrack(i + 1, path + [nums[i]])
        res = []
        backtrack(0, [])
        return res
    # https://leetcode.com/problems/combination-sum/
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # store each of the candidates in pathes until target reached (save path) or passed (backtrack), passing i allows for numbers to be reused
        def backtrack(remaining, path, start):
            if remaining == 0:
                result.append(path)
                return
            if remaining < 0:
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break
                backtrack(remaining - candidates[i], path + [candidates[i]], i)
        candidates.sort()
        result = []
        backtrack(target, [], 0)
        return result
    # https://leetcode.com/problems/permutations/
    def permute(self, nums: List[int]) -> List[List[int]]:
        # go through each num's pathes (with nums except the current one) while storing until we reached a path of nums length (valid permutation)
        def backtrack(path, options):
            if len(path) == len(nums):
                result.append(path)
                return
            for i in range(len(options)):
                backtrack(path + [options[i]], options[:i] + options[i+1:])
        result = []
        backtrack([], nums)
        return result
    # https://leetcode.com/problems/subsets-ii/
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # same as subsets() but with an extra check for the current number to be different than the previous one to avoid duplicates
        def backtrack(start, path):
            res.append(path)
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                backtrack(i + 1, path + [nums[i]])
        nums.sort()
        res = []
        backtrack(0, [])
        return res
    # https://leetcode.com/problems/combination-sum-ii/
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # same as combinationSum() but with an extra check for the current number to be different than the previous one to avoid duplicates
        def backtrack(remaining, path, start):
            if remaining == 0:
                result.append(list(path))
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] > remaining:
                    break
                backtrack(remaining - candidates[i], path + [candidates[i]], i + 1)
        candidates.sort()
        result = []
        backtrack(target, [], 0)
        return result
    # https://leetcode.com/problems/word-search/
    def exist(self, board: List[List[str]], word: str) -> bool:
        # backtrack off each board spot's 4 directions while marking visiteds until index is word length (found) or out of bounds/invalid letter (prune)
        rows, cols = len(board), len(board[0])
        def backtrack(row, col, index):
            if index == len(word):
                return True
            if row < 0 or row >= rows or col < 0 or col >= cols or board[row][col] != word[index]:
                return False
            board[row][col], temp = '#', board[row][col]
            found = (backtrack(row+1, col, index+1) or backtrack(row-1, col, index+1) or
                    backtrack(row, col+1, index+1) or backtrack(row, col-1, index+1))
            board[row][col] = temp
            return found
        return any(backtrack(row, col, 0) for row in range(rows) for col in range(cols))
    # https://leetcode.com/problems/palindrome-partitioning/
    def partition(self, s: str) -> List[List[str]]:
        # backtrack through s until its end while saving path when current substring is equal to invert (palindrome)
        def backtrack(start, path):
            if start == len(s):
                result.append(path)
                return
            for end in range(start+1, len(s)+1):
                if s[start:end] == s[start:end][::-1]:
                    backtrack(end, path + [s[start:end]])
        result = []
        backtrack(0, [])
        return result
    # https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    def letterCombinations(self, digits: str) -> List[str]:
        # backtrack for each of the phone letters mapping of each of the digits until we reached its end
        if not digits:
            return []
        phone_map = {
            "2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
            "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
        }
        def backtrack(index, path):
            if index == len(digits):
                combinations.append(path)
                return
            for letter in phone_map[digits[index]]:
                backtrack(index + 1, path + letter)
        combinations = []
        backtrack(0, "")
        return combinations
    # https://leetcode.com/problems/n-queens/
    def solveNQueens(self, n: int) -> List[List[str]]:
        # backtrack each row trying to place a queen in every column while checking for invalid spots with other queens' cover sets until we reach the last row
        def backtrack(row):
            if row == n:
                result.append(["".join(board[r]) for r in range(n)])
                return
            for col in range(n):
                if col in cols or (row - col) in diag1 or (row + col) in diag2:
                    continue
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)
        result = []
        cols, diag1, diag2 = set(), set(), set()
        board = [['.'] * n for _ in range(n)]
        backtrack(0)
        return result
    
    # Heap / Priority Queue

    # https://leetcode.com/problems/kth-largest-element-in-a-stream/
    class KthLargest:
        # init the heap with test scores (nums) and get k
        def __init__(self, k: int, nums: List[int]):
            self.heap = nums
            heapify(self.heap)
            self.k = k
        # push val in heap, shrink the heap to the largest k elements, return its first/smallest element (therefore the k-th largest)
        def add(self, val: int) -> int:
            heappush(self.heap, val)
            while len(self.heap) > self.k:
                heappop(self.heap)
            return self.heap[0]
    # https://leetcode.com/problems/last-stone-weight/
    def lastStoneWeight(self, stones: list[int]) -> int:
        # use a max-heap (negative/reversed min-heap) to get the biggest weights and substract them if not equal until we reach last stone or none left
        while len(stones) > 1:
            stones.sort(reverse=True)
            first = stones.pop(0)
            second = stones.pop(0)
            if first != second:
                stones.append(first - second)
        return stones[0] if stones else 0
    # https://leetcode.com/problems/k-closest-points-to-origin/
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # push each point and its distance to origin in a k size max-heap
        max_heap = []
        for (x, y) in points:
            dist = -(x * x + y * y)
            heappush(max_heap, (dist, [x, y]))
            if len(max_heap) > k:
                heappop(max_heap)
        return [point for (_, point) in max_heap]
    # https://leetcode.com/problems/kth-largest-element-in-an-array/
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # start a k first nums heap, replace its 1rst for each of the remaining nums if larger so that min_heap[0] is always the k-th largest overall
        min_heap = nums[:k]
        heapify(min_heap)
        for num in nums[k:]:
            if num > min_heap[0]:
                heapreplace(min_heap, num)
        return min_heap[0]
    # https://leetcode.com/problems/task-scheduler/
    def leastInterval(self, tasks: list[str], n: int) -> int:
        # max-heap tasks frequencies and use a cooldowns queue to minimize idle time
        task_counts = Counter(tasks)
        max_heap = [-count for count in task_counts.values()]
        heapify(max_heap)
        cooldown_queue = deque()
        time = 0
        while max_heap or cooldown_queue:
            time += 1
            if max_heap:
                freq = heappop(max_heap)
                if freq + 1 < 0:
                    cooldown_queue.append((freq + 1, time + n))
            if cooldown_queue and cooldown_queue[0][1] == time:
                heappush(max_heap, cooldown_queue.popleft()[0])
        return time
    # https://leetcode.com/problems/design-twitter/
    class Twitter:
        # tweets (userId -> deque of (timestamp, tweetId)), followings (userId -> set of followeeIds) and global timestamp to keep track of tweet order
        def __init__(self):
            self.tweets = defaultdict(deque)
            self.following = defaultdict(set)
            self.timestamp = 0

        # add the tweet to the user's tweet list with current timestamp and increment the global timestamp for the next tweet
        def postTweet(self, userId: int, tweetId: int) -> None:
            self.tweets[userId].appendleft((self.timestamp, tweetId))
            self.timestamp += 1

        # gets the user's then their followings' tweets into a max-heap up to 10 then reverse for desc time order
        def getNewsFeed(self, userId: int) -> list[int]:
            max_heap = []
            if userId in self.tweets:
                for tweet in self.tweets[userId]:
                    heappush(max_heap, (-tweet[0], tweet[1]))
                    if len(max_heap) > 10:
                        heappop(max_heap)
            for followeeId in self.following[userId]:
                if followeeId in self.tweets:
                    for tweet in self.tweets[followeeId]:
                        heappush(max_heap, (-tweet[0], tweet[1]))
                        if len(max_heap) > 10:
                            heappop(max_heap)
            result = []
            while max_heap:
                result.append(heappop(max_heap)[1])
            return result[::-1]

        # add followeeId to followerId's followings if it's not themselves
        def follow(self, followerId: int, followeeId: int) -> None:
            if followerId != followeeId:
                self.following[followerId].add(followeeId)

        # If the follower is following the followee, remove the followee
        def unfollow(self, followerId: int, followeeId: int) -> None:
            if followeeId in self.following[followerId]:
                self.following[followerId].remove(followeeId)
    # https://leetcode.com/problems/find-median-from-data-stream/
    class MedianFinder:
        # init max-heap for the lower half and min-heap for the upper half
        def __init__(self):
            self.low = []
            self.high = []

        # add the num to the appropriate heap (low but high if too large) then balance the heap sizes (low has always either has many nums or 1 more than high)
        def addNum(self, num: int) -> None:
            heappush(self.low, -num)
            if self.low and self.high and (-self.low[0] > self.high[0]):
                heappush(self.high, -heappop(self.low))
            if len(self.low) > len(self.high) + 1:
                heappush(self.high, -heappop(self.low))
            if len(self.high) > len(self.low):
                heappush(self.low, -heappop(self.high))

        # classic median but with the low/high heaps system where their 1rst elements are therefore the middle of a merge
        def findMedian(self) -> float:
            if len(self.low) == len(self.high):
                return (-self.low[0] + self.high[0]) / 2
            else:
                return -self.low[0]
    
    # Graphs

    # https://leetcode.com/problems/number-of-islands/
    def numIslands(self, grid: List[List[str]]) -> int:
        # traverse the grid, each time DFS exploring an entire island and removing it while incrementing of the count
        if not grid:
            return 0
        def DFS(i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
                return
            grid[i][j] = '0'
            DFS(i - 1, j)
            DFS(i + 1, j)
            DFS(i, j - 1)
            DFS(i, j + 1)
        island_count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    island_count += 1
                    DFS(i, j)
        return island_count
    # https://leetcode.com/problems/max-area-of-island/
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # like numIslands() but DFS returns 1 (current) + the area sum of all 4 directions and keep track of the max_area to return it
        if not grid:
            return 0
        def DFS(i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            return 1 + DFS(i - 1, j) + DFS(i + 1, j) + DFS(i, j - 1) + DFS(i, j + 1)
        max_area = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    max_area = max(max_area, DFS(i, j))
        return max_area
    # https://leetcode.com/problems/clone-graph/
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # DFS from the given node and its neighbors recursively until we've clone each
        if not node:
            return None
        cloned_nodes = {}
        def DFS(node: 'Node') -> 'Node':
            if node in cloned_nodes:
                return cloned_nodes[node]
            clone = Node(node.val)
            cloned_nodes[node] = clone
            for neighbor in node.neighbors:
                clone.neighbors.append(DFS(neighbor))
            return clone
        return DFS(node)
    # https://leetcode.com/problems/walls-and-gates/
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        # queue treasure chest positions then BFS all 4 directions from each while updating the valid next positions' distances and adding them to the queue
        if not grid or not grid[0]: return
        row_count, col_count = len(grid), len(grid[0])
        INF = 2147483647
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = deque()
        for row in range(row_count):
            for col in range(col_count):
                if grid[row][col] == 0:
                    queue.append((row, col))
        while queue:
            current_row, current_col = queue.popleft()
            for row_offset, col_offset in directions:
                next_row, next_col = current_row + row_offset, current_col + col_offset
                if 0 <= next_row < row_count and 0 <= next_col < col_count and grid[next_row][next_col] == INF:
                    grid[next_row][next_col] = grid[current_row][current_col] + 1
                    queue.append((next_row, next_col))
    # https://leetcode.com/problems/rotting-oranges/
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # count freshes and queue rotten positions then BFS all 4 directions from each while rotting freshes and adding them to the queue with minute incrementation
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_orange_count = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 2:
                    queue.append((row, col, 0))
                elif grid[row][col] == 1:
                    fresh_orange_count += 1
        minutes_elapsed = 0
        while queue:
            row, col, minutes_elapsed = queue.popleft()
            for d_row, d_col in directions:
                adj_row, adj_col = row + d_row, col + d_col
                if 0 <= adj_row < rows and 0 <= adj_col < cols and grid[adj_row][adj_col] == 1:
                    grid[adj_row][adj_col] = 2
                    fresh_orange_count -= 1
                    queue.append((adj_row, adj_col, minutes_elapsed + 1))
        return minutes_elapsed if fresh_orange_count == 0 else -1
    # https://leetcode.com/problems/pacific-atlantic-water-flow/
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        # DFS from ocean bordering cells to adjacent higher grounds while saving positions for their respective ocean, result is positions that are in both
        if not heights:
            return []
        rows, cols = len(heights), len(heights[0])
        pacific_reachable, atlantic_reachable = set(), set()
        def DFS(row, col, reachable_set, prev_height):
            if (row, col) in reachable_set or row < 0 or row >= rows or col < 0 or col >= cols or heights[row][col] < prev_height:
                return
            reachable_set.add((row, col))
            for delta_row, delta_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                DFS(row + delta_row, col + delta_col, reachable_set, heights[row][col])
        for row in range(rows):
            DFS(row, 0, pacific_reachable, heights[row][0])
            DFS(row, cols - 1, atlantic_reachable, heights[row][cols - 1])
        for col in range(cols):
            DFS(0, col, pacific_reachable, heights[0][col])
            DFS(rows - 1, col, atlantic_reachable, heights[rows - 1][col])
        return list(pacific_reachable & atlantic_reachable)
    # https://leetcode.com/problems/surrounded-regions/
    def solve(self, board: List[List[str]]) -> None:
        # DFS from all border Os to adjacent Os while marking them B, the remaining Os become X as unconnected (also revert Bs to O for valid output)
        if not board:
            return
        row_count, col_count = len(board), len(board[0])
        def DFS(row, col):
            if row < 0 or col < 0 or row >= row_count or col >= col_count or board[row][col] != 'O':
                return
            board[row][col] = 'B'
            for next_row, next_col in [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]:
                DFS(next_row, next_col)
        for row in range(row_count):
            if board[row][0] == 'O':
                DFS(row, 0)
            if board[row][col_count-1] == 'O':
                DFS(row, col_count-1)
        for col in range(col_count):
            if board[0][col] == 'O':
                DFS(0, col)
            if board[row_count-1][col] == 'O':
                DFS(row_count-1, col)
        for row in range(row_count):
            for col in range(col_count):
                board[row][col] = 'X' if board[row][col] == 'O' else ('O' if board[row][col] == 'B' else board[row][col])
    # https://leetcode.com/problems/course-schedule/
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # create adjacency and status lists (0,-1,1 = Unvisited,Visiting,Visited), DFS through each course and stop as soon as a cycle (already visiting) is detected
        adj_list = [[] for _ in range(numCourses)]
        for dest, src in prerequisites:
            adj_list[src].append(dest)
        status = [0] * numCourses
        def DFS(course):
            if status[course] == -1:
                return False
            if status[course] == 1:
                return True
            status[course] = -1
            for neighbor in adj_list[course]:
                if not DFS(neighbor):
                    return False
            status[course] = 1
            return True
        for course in range(numCourses):
            if not DFS(course):
                return False
        return True
    # https://leetcode.com/problems/course-schedule-ii/
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # same as canFinish() but we save visited courses' orders (returned in reverse because of recursion)
        adj_list = [[] for _ in range(numCourses)]
        for dest, src in prerequisites:
            adj_list[src].append(dest)
        status = [0] * numCourses
        course_order = []
        def DFS(course):
            if status[course] == -1:
                return False
            if status[course] == 1:
                return True
            status[course] = -1
            for neighbor in adj_list[course]:
                if not DFS(neighbor):
                    return False
            status[course] = 1
            course_order.append(course)
            return True
        for course in range(numCourses):
            if status[course] == 0:
                if not DFS(course):
                    return []
        return course_order[::-1]
    # https://leetcode.com/problems/valid-tree/
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # create adjacency list, DFS from its node 0 while storing uniques, n unique nodes=valid graph (less=not connected, more=cycles, skip parents false cycles)
        if len(edges) != n - 1:
            return False
        adj_list = {i: [] for i in range(n)}
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        visited = set()
        def DFS(node, parent):
            visited.add(node)
            for neighbor in adj_list[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return False
                if not DFS(neighbor, node):
                    return False
            return True
        if not DFS(0, -1):
            return False
        return len(visited) == n
    # https://leetcode.com/problems/count-connected-components/
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # create adjacency list, DFS from each of its non-visited nodes while incrementing the count and marking visiteds
        adj_list = {i: [] for i in range(n)}
        for a, b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        def dfs(node):
            visited.add(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        visited = set()
        component_count = 0
        for node in range(n):
            if node not in visited:
                dfs(node)
                component_count += 1
        return component_count
    # https://leetcode.com/problems/redundant-connection/
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # use a graph dict and visiteds set to DFS through each edge and check if it creates a cycle (already visited or same edge)
        graph = defaultdict(list)
        def DFS(node, target, visited):
            if node == target:
                return True
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if DFS(neighbor, target, visited):
                        return True
            return False
        for u, v in edges:
            visited = set()
            if u in graph and v in graph and DFS(u, v, visited):
                return [u, v]
            graph[u].append(v)
            graph[v].append(u)
        return []
    # https://leetcode.com/problems/word-ladder/
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # set for quick lookup, queue for words to explore+trans len, BFS queue by trying char swaps to get valid next words and find the shortest trans sequence
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0
        queue = deque([(beginWord, 1)])
        while queue:
            current_word, length = queue.popleft()
            for i in range(len(current_word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = current_word[:i] + c + current_word[i+1:]
                    if next_word == endWord:
                        return length + 1
                    if next_word in wordSet:
                        queue.append((next_word, length + 1))
                        wordSet.remove(next_word)
        return 0
    
    # 1-D Dynamic Programming

    # https://leetcode.com/problems/climbing-stairs/
    def climbStairs(self, n: int) -> int:
        # number of ways to reach a step is the sum of the ways to reach the previous 2 steps so calculate the Fibonacci sequence from base case 3 to n
        if n == 1:
            return 1
        elif n == 2:
            return 2
        first = 1
        second = 2
        for _ in range(3, n + 1):
            third = first + second
            first = second
            second = third
        return second
    # https://leetcode.com/problems/min-cost-climbing-stairs/
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # get each step's shortest path being its cost + the min of both previous ones, once done we get the min of the last or 2nd-last step
        n = len(cost)
        if n == 2:
            return min(cost)
        dp = [0] * n
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, n):
            dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
        return min(dp[-1], dp[-2])
    # https://leetcode.com/problems/house-robber/
    def rob(self, nums: List[int]) -> int:
        # get each house's max amount being the max of the previous vs current + 2nd previous as adjacents are a no-go, last element is then total max amount
        if not nums:
            return 0
        n = len(nums)
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
        return dp[-1]
    # https://leetcode.com/problems/house-robber-ii/
    def rob(self, nums: List[int]) -> int:
        # just get the max of rob() on first to second-last vs on second to last since we only need to avoid first and last now being adjacent
        def basic_rob(houses):
            n = len(houses)
            if n == 1:
                return houses[0]
            dp = [0] * n
            dp[0] = houses[0]
            dp[1] = max(houses[0], houses[1])
            for i in range(2, n):
                dp[i] = max(dp[i - 1], houses[i] + dp[i - 2])
            return dp[-1]
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        return max(basic_rob(nums[1:]), basic_rob(nums[:-1]))
    # https://leetcode.com/problems/longest-palindromic-substring/
    def longestPalindrome(self, s: str) -> str:
        # search for each odd or even length palindrome's center accross s, when found and longest save its start and length for substring return
        if s == s[::-1]:
            return s
        longest_start = 0
        longest_size = 1
        for center in range(1, len(s)):
            left_bound_odd = center - longest_size - 1
            left_bound_even = center - longest_size
            right_bound = center + 1
            odd_palindrome = s[left_bound_odd:right_bound]
            if left_bound_odd >= 0 and odd_palindrome == odd_palindrome[::-1]:
                longest_start = left_bound_odd
                longest_size += 2
            else:
                even_palindrome = s[left_bound_even:right_bound]
                if even_palindrome == even_palindrome[::-1]:
                    longest_start = left_bound_even
                    longest_size += 1
        return s[longest_start: longest_start + longest_size]
    # https://leetcode.com/problems/palindromic-substrings/
    def countSubstrings(self, s: str) -> int:
        # advance palindrome center i through s, for each 1rst get same consecutive chars then different ones with left and right pointers, updated result is the count
        result = 0
        i = 0
        while i < len(s):
            consecutiveSameChars = 1
            left = i - 1
            while i < len(s) - 1 and s[i] == s[i + 1]:
                i += 1
                consecutiveSameChars += 1
            result += consecutiveSameChars * (consecutiveSameChars + 1) // 2
            right = i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                result += 1
                left -= 1
                right += 1
            i += 1
        return result
    # https://leetcode.com/problems/decode-ways/
    def numDecodings(self, s: str) -> int:
        # traverse s, each time tracking the number of decodings whenever a new valid number is found on the previous 2 (1-9 digits, 10-26 numbers)
        if not s or s[0] == '0':
            return 0
        n = len(s)
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            if 1 <= int(s[i-1:i]) <= 9:
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[n]
    # https://leetcode.com/problems/coin-change/
    def coinChange(self, coins: List[int], amount: int) -> int:
        # update an amount-long array of infinites with the fewest coins needed for each, if amount's is still infinite then no solution
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1
    # https://leetcode.com/problems/maximum-product-subarray/
    def maxProduct(self, nums: List[int]) -> int:
        # simple store of the max product along nums except we also store the min to swap when negative so that we keep actual max/min products
        if not nums:
            return 0
        max_product = nums[0]
        min_product = nums[0]
        result = nums[0]
        for i in range(1, len(nums)):
            num = nums[i]
            if num < 0:
                max_product, min_product = min_product, max_product
            max_product = max(num, max_product * num)
            min_product = min(num, min_product * num)
            result = max(result, max_product)
        return result
    # https://leetcode.com/problems/word-break/
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # init s-long falses array, dp[i] true if substring s[0:i] in dict and previous is true too (both valid segments), if all true then valid segmentation
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]
    # https://leetcode.com/problems/longest-increasing-subsequence/
    def lengthOfLIS(self, nums: List[int]) -> int:
        # build the LIS through nums, each should either replace LIS' head/tail using left binary search's index or just be appended, return LIS' len
        sub = []
        for num in nums:
            i = bisect_left(sub, num)
            if i < len(sub):
                sub[i] = num 
            else:
                sub.append(num)
        return len(sub)
    # https://leetcode.com/problems/partition-equal-subset-sum/
    def canPartition(self, nums: List[int]) -> bool:
        # odd total=not possible, update (backward to avoid reusing numbers) target-long falses array where dp[i]=subset with sum i exists so true dp[target]=good
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for num in nums:
            for i in range(target, num - 1, -1):
                dp[i] = dp[i] or dp[i - num]
        return dp[target]
    
    # Intervals

    # https://leetcode.com/problems/insert-interval/
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # build a new intervals array by appending, in order, either current intervals or newIntervals if they dont overlap or a merge of both if they do
        result = []
        for interval in intervals:
            if interval[1] < newInterval[0]:
                result.append(interval)
            elif interval[0] > newInterval[1]:
                result.append(newInterval)
                newInterval = interval
            else:
                newInterval[0] = min(newInterval[0], interval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        result.append(newInterval)
        return result
    # https://leetcode.com/problems/merge-intervals/
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # sort intervals by start time, build a new intervals array by either appending if first/no overlap or merging by updating the end time
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
    # https://leetcode.com/problems/non-overlapping-intervals/
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # sort intervals by end to check overlaps with the latest valid interval by start while counting them
        intervals.sort(key=lambda x: x[1])
        end = intervals[0][1]
        count = 0
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                count += 1
            else:
                end = intervals[i][1]
        return count
    # https://neetcode.io/problems/meeting-schedule/
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        # sort intervals by start to check overlaps with ends and return false if so
        intervals.sort(key=lambda x: x[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True
    # https://neetcode.io/problems/meeting-schedule-ii/
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        # sort intervals by start time to track overlaps with the updated earliest ending meeting in a min-heap, it contains min required days at the end
        intervals.sort(key=lambda x: x.start)
        min_heap = []
        for interval in intervals:
            if min_heap and min_heap[0] <= interval.start:
                heappop(min_heap)
            heappush(min_heap, interval.end)
        return len(min_heap)
    # https://leetcode.com/problems/minimum-interval-to-include-each-query/
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        # sort intervals by start to track valid ones in a min-heap for each query, res then maps the shortest ones to a valid return
        intervals.sort(key = lambda x: x[0])
        minHeap = []
        res = {}
        i = 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                heappush(minHeap, (intervals[i][1] - intervals[i][0] + 1, intervals[i][1]))
                i += 1
            while minHeap and minHeap[0][1] < q:
                heappop(minHeap)
            res[q] = minHeap[0][0] if minHeap else -1
        return [ res[q] for q in queries]
    
    # Greedy

    # https://leetcode.com/problems/maximum-subarray/
    def maxSubArray(self, nums: List[int]) -> int:
        # iterate through nums while keeping track of the local subarray's sum and the maximum sum of any subarray which we'll return
        max_sum = current_sum = nums[0]
        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum
    # https://leetcode.com/problems/jump-game/
    def canJump(self, nums: List[int]) -> bool:
        # iterate backward from 2nd-last num while updating the min required jumps to reach the end from each position, if still 0 at the end then it's possible
        require = 0
        for i in reversed(nums[:-1]):
            if i > require:
                require = 0
            else:
                require += 1
        return not require
    # https://leetcode.com/problems/jump-game-ii/
    def jump(self, nums: List[int]) -> int:
        # iterate nums while saving each jump's end and the current farthest reach until we reach the end, return the jumps count incremented at each jump's end
        jumps = 0
        current_end = 0
        farthest = 0
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            if i == current_end:
                jumps += 1
                current_end = farthest
                if current_end >= len(nums) - 1:
                    break
        return jumps
    # https://leetcode.com/problems/gas-station/
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # if negative total (gas - cost) then impossible, otherwise check each gas station's total until is positive and becomes the starting point
        if sum(gas) < sum(cost):
            return -1
        res = 0
        total = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if total < 0:
                total = 0
                res = i + 1
        return res
    # https://leetcode.com/problems/hand-of-straights/
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        # if not divisible by groupsize then impossible, count cards frequencies and sort them to create groups for each, if not enough cards for any then impossible
        if len(hand) % groupSize != 0:
            return False
        count = Counter(hand)
        for card in sorted(count):
            if count[card] > 0:
                num_groups_needed = count[card]
                for i in range(groupSize):
                    if count[card + i] < num_groups_needed:
                        return False
                    count[card + i] -= num_groups_needed
        return True
    # https://leetcode.com/problems/merge-triplets-to-form-target-triplet/
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        # for each triplets' numbers check for respective target equality (or lower as we can max() it), if all 3 target flags are true then possible
        x, y, z = target
        found_x = found_y = found_z = False
        for a, b, c in triplets:
            if a <= x and b <= y and c <= z:
                if a == x:
                    found_x = True
                if b == y:
                    found_y = True
                if c == z:
                    found_z = True
            if found_x and found_y and found_z:
                return True
        return False
    # https://leetcode.com/problems/partition-labels/
    def partitionLabels(self, s: str) -> List[int]:
        # record occurence of last characters, for each char update partition end with larger occurence and when reached save the partition for the return
        last_occurrence = {char: idx for idx, char in enumerate(s)}
        partitions = []
        start, end = 0, 0
        for i, char in enumerate(s):
            end = max(end, last_occurrence[char])
            if i == end:
                partitions.append(end - start + 1)
                start = i + 1
        return partitions
    # https://leetcode.com/problems/valid-parenthesis-string/
    def checkValidString(self, s: str) -> bool:
        # track min and max number of unmatched left parentheses (low, high), if high ever negative then too many ')', if low 0 by the end then valid
        low, high = 0, 0
        for char in s:
            if char == '(':
                low += 1
                high += 1
            elif char == ')':
                low -= 1
                high -= 1
            elif char == '*':
                low -= 1
                high += 1
            if high < 0:
                return False
            low = max(low, 0)
        return low == 0
    
    # Advanced Graphs

    # https://leetcode.com/problems/reconstruct-itinerary/
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # lexically sort tickets and setup deque for efficient pops by DFSing through the graph to construct the itinerary from JFK, reverse return for correct order
        graph = defaultdict(deque)
        tickets.sort()
        for departure, arrival in tickets:
            graph[departure].append(arrival)
        res = []
        def dfs(start):
            while graph[start]:
                next_destination = graph[start].popleft()
                dfs(next_destination)
            res.append(start)
        dfs('JFK')
        return res[::-1]
    # https://leetcode.com/problems/min-cost-to-connect-all-points/
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # (dist,point) min-heap and visited points set to, for each point, get unvisited edges' dists after summing the nearest one into the returned minimum total
        n = len(points)
        if n == 1:
            return 0
        min_heap = [(0, 0)]
        total_cost = 0
        visited = set()
        while len(visited) < n:
            dist, point = heappop(min_heap)
            if point in visited:
                continue
            total_cost += dist
            visited.add(point)
            for neighbor in range(n):
                if neighbor not in visited:
                    dist = abs(points[point][0] - points[neighbor][0]) + abs(points[point][1] - points[neighbor][1])
                    heappush(min_heap, (dist, neighbor))
        return total_cost
    # https://leetcode.com/problems/network-delay-time/
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Djikstra's with (weight, node) min-heap, unvisiteds set and inf-initialized weights, if there's still any by the end then impossible
        edges = defaultdict(dict)
        for source, dest, weight in times:
            edges[source][dest] = weight
        unvisited = set(range(1, n + 1))
        weights = [float('inf')] * (n + 1)
        weights[k] = 0
        queue = [(0, k)]
        while queue and unvisited:
            cur_dist, node = heappop(queue)
            if node not in unvisited:
                continue
            unvisited.remove(node)
            for neighbor, travel_time in edges[node].items():
                new_dist = cur_dist + travel_time
                if new_dist < weights[neighbor]:
                    weights[neighbor] = new_dist
                    heappush(queue, (new_dist, neighbor))
        max_distance = max(weights[1:])
        return max_distance if max_distance < float('inf') else -1
    # https://leetcode.com/problems/swim-in-rising-water/
    def swimInWater(self, grid: List[List[int]]) -> int:
        # go from top left through each cell and its within-bounds unvisited neighbors while tracking the lowest's max time until we reach bottom right
        n = len(grid)
        min_heap = [(grid[0][0], 0, 0)]
        visited = set((0, 0))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        max_time = 0
        while min_heap:
            time, x, y = heappop(min_heap)
            max_time = max(max_time, time)
            if (x, y) == (n - 1, n - 1):
                return max_time
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < n and 0 <= new_y < n and (new_x, new_y) not in visited:
                    visited.add((new_x, new_y))
                    heappush(min_heap, (grid[new_x][new_y], new_x, new_y))
    # https://leetcode.com/problems/foreign-dictionary/
    def foreignDictionary(self, words: List[str]) -> str:
        # build graph/in-degrees, for each word pair add first differing chars' edge, detect invalid prefix, topologically sort zero in-degree nodes, return order
        graph = defaultdict(set)
        in_degree = {char: 0 for word in words for char in word}
        for i in range(len(words) - 1):
            first, second = words[i], words[i + 1]
            for c1, c2 in zip(first, second):
                if c1 != c2:
                    if c2 not in graph[c1]:
                        graph[c1].add(c2)
                        in_degree[c2] += 1
                    break
            else:
                if len(second) < len(first):
                    return ""
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        order = []
        while queue:
            char = queue.popleft()
            order.append(char)
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return "".join(order) if len(order) == len(in_degree) else ""
    # https://leetcode.com/problems/cheapest-flights-within-k-stops/
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # go through each closest city's neighbors (cost, city, remaining stops min-heap) from src until we reach dest while tracking visited stops to be under k
        graph = defaultdict(dict)
        for start, destination, cost in flights:
            graph[start][destination] = cost
        min_heap = [(0, src, k+1)]
        visited_stops = [0] * n
        while min_heap:
            current_cost, current_city, remaining_stops = heappop(min_heap)
            if current_city == dst:
                return current_cost
            if visited_stops[current_city] >= remaining_stops:
                continue
            visited_stops[current_city] = remaining_stops
            for neighbor_city, flight_cost in graph[current_city].items():
                heappush(min_heap, (current_cost + flight_cost, neighbor_city, remaining_stops - 1))
        return -1
    
    # 2-D Dynamic Programming

    # https://leetcode.com/problems/unique-paths/
    def uniquePaths(self, m: int, n: int) -> int:
        # robot has to move (down m-1 & right n-1 times)=(m−1)+(n−1)=m+n−2, binomial coefficient with m-1 gives how many diff ways we can go down on total
        return comb(m + n - 2, m - 1)
    # https://leetcode.com/problems/longest-common-subsequence/
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # fill a (len(text1)+1)*(len(text2)+1) matrix where dp[i][j]=LCS at text1[:i] and text2[:j] (either increment or max) until bottom right is the total LCS
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[len(text1)][len(text2)]
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
    def maxProfit(self, prices: List[int]) -> int:
        # init and build max profits for each day/action with their formula based on their previous day, result is max profit on last day without holding any stock
        if not prices:
            return 0
        n = len(prices)
        hold, sell, cooldown = [0] * n, [0] * n, [0] * n
        hold[0] = -prices[0]
        sell[0] = 0
        cooldown[0] = 0
        for i in range(1, n):
            hold[i] = max(hold[i-1], cooldown[i-1] - prices[i])
            sell[i] = hold[i-1] + prices[i]
            cooldown[i] = max(cooldown[i-1], sell[i-1])
        return max(sell[-1], cooldown[-1])
    # https://leetcode.com/problems/coin-change-ii/
    def change(self, amount: int, coins: List[int]) -> int:
        # fill array where dp[i]=number of ways to make amount i based on each coin
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[amount]
    # https://leetcode.com/problems/target-sum/
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # P sum of + nums, N sum of - nums, P−N=target and P+N=total_sum so P=(target+total_sum)/2 (target sum), dp[i]=ways to reach sum i so dp[P] is the answer
        total_sum = sum(nums)
        if (total_sum + target) % 2 != 0 or total_sum < abs(target):
            return 0
        P = (total_sum + target) // 2
        dp = [0] * (P + 1)
        dp[0] = 1
        for num in nums:
            for i in range(P, num - 1, -1):
                dp[i] += dp[i - num]
        return dp[P]
    # https://leetcode.com/problems/interleaving-string/
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # dp[j]=can s1[:i]&s2[:j] make s3[:i+j], check if s2 then s1 then both's chars can continue forming s3 until i+j-1 so that dp[len(s2)]=result
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [False] * (len(s2) + 1)
        dp[0] = True
        for j in range(1, len(s2) + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
        for i in range(1, len(s1) + 1):
            dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
            for j in range(1, len(s2) + 1):
                dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or (dp[j - 1] and s2[j - 1] == s3[i + j - 1])
        return dp[len(s2)]
    # https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # dp[i][j]=LIP from i/j, iterate value-sorted cells to check each in-bound direction if we can move to a smaller value, max dp is LIP
        if not matrix:
            return 0
        rows, cols = len(matrix), len(matrix[0])
        dp = [[1] * cols for _ in range(rows)]
        cells = [(matrix[i][j], i, j) for i in range(rows) for j in range(cols)]
        cells.sort()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for value, i, j in cells:
            for dx, dy in directions:
                x, y = i + dx, j + dy
                if 0 <= x < rows and 0 <= y < cols and matrix[x][y] < matrix[i][j]:
                    dp[i][j] = max(dp[i][j], dp[x][y] + 1)
        return max(max(row) for row in dp)
    # https://leetcode.com/problems/distinct-subsequences/
    def numDistinct(self, s: str, t: str) -> int:
        # dp[i][j]=nb of s[:i] subsequences that equal t[:j], for each s/t prefixes update dp if char matches (match+ignore) or not (ignore), result is last
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m][n]
    # https://leetcode.com/problems/edit-distance/
    def minDistance(self, word1: str, word2: str) -> int:
        # dp[i][j] = min operations to convert word1[:i] into word2[:j], for each word1/word2 chars if diff char pick min cost else skip, result is last
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i][j - 1],  # Insert
                        dp[i - 1][j],  # Delete
                        dp[i - 1][j - 1]  # Replace
                    ) + 1
        return dp[m][n]
    # https://leetcode.com/problems/burst-balloons/
    def maxCoins(self, nums: List[int]) -> int:
        # dp[left][right]=max coins bursting in range [left, right], for each range get max coins of its last balloon, result is last (with edges' 1  paddings)
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for length in range(1, n - 1):
            for left in range(1, n - length):
                right = left + length - 1
                for lastBurst in range(left, right + 1):
                    dp[left][right] = max(
                        dp[left][right],
                        dp[left][lastBurst-1] + nums[left-1] * nums[lastBurst] * nums[right+1] + dp[lastBurst+1][right]
                    )
        return dp[1][n-2]
    # https://leetcode.com/problems/regular-expression-matching/
    def isMatch(self, s: str, p: str) -> bool:
        # dp[i][j]=does s[:i] match p[:j], init empty string patterns, for each chars fill dp based on same chars/"." or "*", result is last (entire string match)
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (p[j - 2] == '.' or p[j - 2] == s[i - 1]))
        return dp[m][n]

    # Bit Manipulation

    # https://leetcode.com/problems/single-number/
    def singleNumber(self, nums: List[int]) -> int:
        # any number XORed with itself is 0 and any number XORed with 0 remains unchanged so duplicates cancel each other out leaving the single num at the end
        result = 0
        for num in nums:
            result ^= num
        return result
    # https://leetcode.com/problems/number-of-1-bits/
    def hammingWeight(self, n: int) -> int:
        # right shift through n, incrementing the returned count everytime it's 1 with AND
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    # https://leetcode.com/problems/counting-bits/
    def countBits(self, n: int) -> List[int]:
        # fill result dp array with dp[i>>1]=nb of 1s in i without LSB (i>>1 equivalent to i//2) + i&1=1 if the LSB of i is 1 (if even i//2 1s, if odd i//2+1 1s)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp
    # https://leetcode.com/problems/reverse-bits/
    def reverseBits(self, n: int) -> int:
        # for each last bit, extract(AND)/place(OR) it at n's start by left-shifting it at its reversed spot before deleting its old last spot by right-shifting
        result = 0
        for i in range(32):
            result |= (n & 1) << (31 - i)
            n >>= 1
        return result
    # https://leetcode.com/problems/missing-number/
    def missingNumber(self, nums: List[int]) -> int:
        # get the XOR of nums with and without the missing number, given XOR's properties (see singleNumber()), the XOR of both is the missing number
        xor_all = 0
        xor_nums = 0
        for i in range(len(nums) + 1):
            xor_all ^= i
        for num in nums:
            xor_nums ^= num
        return xor_all ^ xor_nums
    # https://leetcode.com/problems/sum-of-two-integers/
    def getSum(self, a: int, b: int) -> int:
        # while b (carry, AND << 1) isn't done, add it to a (sum, XOR), while masking both to simulate 32-bit overflow, if result exceeds INT_MAX then negative overflow
        while b != 0:
            sum_without_carry = a ^ b
            carry = (a & b) << 1
            a = sum_without_carry & 0xFFFFFFFF
            b = carry & 0xFFFFFFFF
        return a if a <= 2147483647 else ~(a ^ 0xFFFFFFFF)
    # https://leetcode.com/problems/reverse-integer/
    def reverse(self, x: int) -> int:
        # extract each absoluted x's last digit into result*10 (classic reversal) before giving back original sign and returning 0 if overflow, not bit manipulation
        result = 0
        sign = -1 if x < 0 else 1
        x = abs(x)
        while x:
            result = result * 10 + x % 10
            x //= 10
        result *= sign
        return 0 if result > 2147483647 or result < -2147483648 else result
    
    # Math & Geometry

    # https://leetcode.com/problems/rotate-image/
    def rotate(self, matrix: List[List[int]]) -> None:
        # transpose (swap rows with columns) matrix then inverse its rows to rotate matrix by 90° clockwise
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()
    # https://leetcode.com/problems/spiral-matrix/
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # while matrix isnt empty, pop row then transpose (zip) the remaining matrix and reverse ([::-1]) it to simulate counter-clockwise rotation for spiral
        result = []
        while matrix:
            result += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return result
    # https://leetcode.com/problems/set-matrix-zeroes/
    def setZeroes(self, matrix: List[List[int]]) -> None:
        # use 1rst row/column as i/j axes to: 0-mark matrix[i][j] 0s->0 out line cells based on markers->0 out 1rst row/column if needed
        rows, cols = len(matrix), len(matrix[0])
        first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
        first_col_zero = any(matrix[i][0] == 0 for i in range(rows))
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if first_row_zero:
            for j in range(cols):
                matrix[0][j] = 0
        if first_col_zero:
            for i in range(rows):
                matrix[i][0] = 0
    # https://leetcode.com/problems/happy-number/
    def isHappy(self, n: int) -> bool:
        # simply replace the number by the sum of the squares of its digits while saving it in a set until it becomes 1 (true) or already appeared in the set (false)
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = sum(int(digit) ** 2 for digit in str(n))
        return n == 1
    # https://leetcode.com/problems/plus-one/
    def plusOne(self, digits: List[int]) -> List[int]:
        # for each digit starting from end, if under 9 increment and return digits, else it's 0 and continue (carry 1 to next), if all needed carry return with 1 prepend
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            digits[i] = 0
        return [1] + digits
    # https://leetcode.com/problems/powx-n/
    def myPow(self, x: float, n: int) -> float:
        # Divide and Conquer using formulas: xⁿ⁽ᵉᵛᵉⁿ⁾=(xⁿᐟ²)², xⁿ⁽ᵒᵈᵈ⁾=x*xⁿ⁻¹, x⁻ⁿ=1/xⁿ
        if n == 0:
            return 1
        if n < 0:
            x = 1 / x
            n = -n
        result = 1
        while n != 0:
            if n % 2 == 1:
                result *= x
            x *= x
            n //= 2
        return result
    # https://leetcode.com/problems/multiply-strings/
    def multiply(self, num1: str, num2: str) -> str:
        # convert num1 num2 to integers, multiply, convert them back to strings, use str:int dict since not allowed to convert to int directly, return result or 0
        num_dict = {str(i): i for i in range(10)}
        n1 = n2 = 0
        for i in num1:
            n1 = n1 * 10 + num_dict[i]
        for i in num2:
            n2 = n2 * 10 + num_dict[i]
        res = n1 * n2
        strr = ""
        while res:
            strr = str(res % 10) + strr
            res //= 10
        return strr or "0"
    # https://leetcode.com/problems/detect-squares/
    class DetectSquares:
        # point (x, y coordinates): frequency (how many times this exact point has been added, default 0) dictionary
        def __init__(self):
            self.point_counts = defaultdict(int)
        # increase frequency of point (array to tuple because hashable for dictionary)
        def add(self, point):
            self.point_counts[tuple(point)] += 1
        # check each stored point if same x&y dist (square) as query (+ diff coords), update square count based on 2 other points with same y as stored/x as query
        def count(self, query_point):
            qx, qy = query_point
            square_count = 0
            for (sx, sy), freq in self.point_counts.items():
                if abs(sx - qx) == abs(sy - qy) and sx != qx and sy != qy:
                    square_count += freq * self.point_counts.get((sx, qy), 0) * self.point_counts.get((qx, sy), 0)
            return square_count