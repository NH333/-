'''我的想法，没AC过'''
'''太关注[]空集合了，忘记直接设置一个初值0就可以解决了'''
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        for i in range(len(nums)-1):
            if nums[:i] == [] and sum(nums[i+1:]) == 0:
                return i
            else:
                if sum(nums[:i]) == sum(nums[i+1:]):
                    return i
        if nums[len(nums):] == [] and sum(nums[:len(nums)-1]) == 0:
            return len(nums) - 1
        return -1


'''官网解答'''
'''enumerate的用法
    左边和设置为0
    右边和先设置为全部和
'''
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return -1
        
        rightsum = sum(nums)
        
        leftsum = 0
        for i, n in enumerate(nums):
            if i > 0:
                leftsum += nums[i-1]  #左边和更新
                
            rightsum -= n   #右边和更新
                
            if leftsum == rightsum:
                return i
                
            print(leftsum, rightsum)
            
            
        return -1