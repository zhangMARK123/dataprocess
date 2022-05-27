def sort(nums):
    def algorithm(time=0):
        if time==length:
            out.append(nums[:])
        for i in range(time,length):
            nums[i],nums[time]=nums[time],nums[i]
            algorithm(time+1)
            nums[i],nums[time]=nums[time],nums[i]
    out=[]
    length=len(nums)
    if length==0 : return []
    algorithm(0)
    return out 
nums=[1,3,4]
print(sort(nums))