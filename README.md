# Perceptron Network Homework
- 系級：資管四乙
- 姓名：郭鎮源
- 學號：405402390
- Learning Rule
    - W(new) = W(old) + eP(T)
    - B(new) = B(old) + e
    - learning rate : 1
    - 最大 Epochs : 1000
    - Activate function : hard limit
- Dataset 1 :
    - Class : 1,2,3,4
    - 當分辨不出來會顯示 No
- Dataset 2 :
    - Class : W,P,O,B
    - 當分辨不出來會顯示 No
- Group
    - 對應至 Class 的 Array


## (a)Two-neuron perceptron.py
- Run : python3 a_Two-neuron_(1)(2)(3)
- (a)Two-neuron perceptron:
    - (1)Dataset 1
        - Class : 1,2,3,4
        - Group : [0, 0], [0, 1], [1, 0], [1, 1]
        - Weight : [[-2  0],[ 0 -2]]
        - Bias : [[-1 ],[ 0]]
        - 分析：發現這個的 Weight 與 Bias 非常好，因為這個的訓練資料太過少，無法非常精準的判別，所以若用其他的 Weigh or Bias 可能會導致影響其測試資料的結果，導致判斷非常不準確，常常會判斷錯誤或是判斷不出來，但是相較於四個神經元來比較，訓練時間較快，且比較準確
    - (2)Dataset 2 – Use the first two components.
        - 執行時間需要等一段時間
        - Class : W,P,O,B
        - Group : [0, 0], [1, 0], [0, 1], [1, 1]
        - Weight : [[0. 0. 0.], [0. 0. 0.] ]
        - Bias : [[0 ],[ 0]]
        - 分析：由於只有兩個component，所以無法將資料區完全的分開來，導致 Error 的值一直無法為0，讓訓練無窮的一直跑下去，無法有效率的辨識出相對應的答案，但不會像四個神經元一樣，不會讓有些的測試資料無法辨識不出結果，所以相對於四個神經元來說，比較好一點
    - (3)Dataset 2 – Use the three components.
        - Class : W,P,O,B
        - Group : [0, 0], [1, 0], [0, 1], [1, 1]
        - Weight : [[0. 0. 0.], [0. 0. 0.] ]
        - Bias : [[0 ],[ 0]]
        - 分析：發現此辨識效率比起兩個component 和四個神經元來說，是效率最好的，因為在裡面有三 component，讓整體的辨識上升許多，不會讓 Error 的值一直無法為0，且與四個神經元比，由於只需用兩條線區分四個Class，不像四個神經元 - 要用四條線區分，導致會有一些模糊地帶，導致無法辨識完全，所以在我的實驗中，此個是在Dataset 2 最佳的選擇

## (b)Four-neuron perceptron.py
- Run : python3 b_Four-neuron_(1)(2)(3)
- (b)Four-neuron perceptron:
    - (1)Dataset 1
        - Class : 1,2,3,4
        - Group : [1, 0, 0, 0], [0, 1, 0, 0 ], [0, 0, 1, 0 ], [0, 0, 0, 1 ]
        - Weight : [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.] ]
        - Bias : [[0 ],[ 0],[ 0],[ 0]]
        - 分析：發現因為是四個神經元，由於訓練數量相對應的少很多，所以無法訓練至相當的精確，且可能是因為我的 Weight 與 Bias 未拿捏好，導致有些的測試資料無法辨識出來相對應的結果，整體來說，我覺得用兩個神經元是最剛好這個Dataset
    - (2)Dataset 2 – Use the first two components.
        - 執行時間需要等一段時間
        - Class : W,P,O,B
        - Group : [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]
        - Weight : [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.] ]
        - Bias : [[0 ],[ 0],[ 0],[ 0]]
        - 分析：發現因為是四個神經元，且由於只有兩個component，將資料區分開來的狀況不是非常好，Error 的值一直無法為0，導致會無窮的一直跑下去，所以會跑到1000次，導致無法有效率的辨識出相對應的答案，讓有些的測試資料無法辨識出結果，所以由此可知這不是非常理想的結果
    - (3)Dataset 2 – Use the three components.
        - Class : W,P,O,B
        - Group : [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]
        - Weight : [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.] ]
        - Bias : [[0 ],[ 0],[ 0],[ 0]]
        - 分析：用四個神經元下去訓練，且用三個component，效果比起兩個component好太多，區分資料的特性好上太多，不會讓 Error 值跑不完，可以讓整個的過程順利的完成，且無兩個component之問題，辨識不出結果的問題，但相對應兩個神經元來比較，需要多一點的 Epoche 來跑完整個的訓練，且若給予不同的 Weight or Bias 會導致訓練過程變得更加久與可能會無窮的一直跑下去．
## Environment
- need to install `numpy` and `pandas`