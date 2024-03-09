## training前需要先建立pytorch環境  
### 1.在開始輸入nvidia 進入nvidia控制面板點開系統資訊在對應欄位可以看到電腦所需的CUDA版本，電腦的CUDA支持版本，是向下兼容的，也就是我們可以安裝<=我们電腦支持版本的CUDA。
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/946aafc3-5ffd-4af3-bb05-4ff6bd5854d4)  
### 2.進入連結下載CUDA: https://developer.nvidia.com/cuda-toolkit-archive  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/e41c5f6b-8fb4-4546-85ba-fc4ccb93e5f6)  




"可能遇到多線程的問題"(訓練模型時如果遇到workers的問題加入以下程式碼在train.py裡面主程式的部分便可以解決):  
解決方式為在主程式前面加上，os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
如果遇到其他沒有安裝的套件就pip install 目標套件  這樣及可以順利安裝  
