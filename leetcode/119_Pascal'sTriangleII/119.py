class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        tmp_up = [1,1]
        tmp_down = []
        tmp_num = [1]
        count = 0
        if rowIndex == 0:
            return [1]
        elif rowIndex < 2:
            return [1]*(rowIndex+1)     
        else:
            for i in range(2,rowIndex+1):
                NeedAppend_Num = i - 1
                for j in range(len(tmp_up)):
                    tmp_num.append(tmp_up[j]+tmp_up[j+1])
                    count += 1
                    if count == NeedAppend_Num:
                        break
                count = 0
                tmp_num.append(1)
                tmp_down = tmp_num.copy()
                tmp_num = [1]
                tmp_up = tmp_down.copy()
            return tmp_down