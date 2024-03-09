## training前需要先建立pytorch環境  
### 1.在開始輸入nvidia 進入nvidia控制面板點開系統資訊在對應欄位可以看到電腦所需的CUDA版本，電腦的CUDA支持版本，是向下兼容的，也就是我們可以安裝<=我们電腦支持版本的CUDA。
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/946aafc3-5ffd-4af3-bb05-4ff6bd5854d4)  
### 2.進入連結下載CUDA: https://developer.nvidia.com/cuda-toolkit-archive  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/e41c5f6b-8fb4-4546-85ba-fc4ccb93e5f6)  
### 3. 下載CUDA步驟。  
依照以下順序點。
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/71d52a2d-c222-497d-bc4f-52299d4b63bb)  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/7b9b9be6-7c68-4e35-87a0-06b1cbb8bd40)  
### 4. 新增剛剛下載的CUDA至環境變數中  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/51d21504-8aa6-4bde-81d3-e6e7ee1d3a3e)  
安裝成功可以至cmd視窗觀看有沒有成功下載。
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/6ae6fd79-994e-4378-846e-6efa41ba8f7a)  
### 5. 至以下網址觀察自己需要的torch 、 torchvision的版本: https://blog.csdn.net/shiwanghualuo/article/details/122860521  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/11c66a41-5df6-40cc-8ad2-13bfe510e4f3)

### 6. 至以下的網址下載對應的torch 以及 torchvision : https://download.pytorch.org/whl/torch_stable.html 要下載cu開頭的代表是cuda版本的，以及cp要是自己python的版本，系統挑選windows。  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/9fb43a19-d46e-4d17-973b-0ebc2c352398) 

### 7. 接續安裝剛剛下載的torch, torchvision  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/42734b67-8224-4b7d-ad58-f3ef954d2d63)  

### 8. 有可能會遇到python版本不對安裝不起來的問題像是如下圖。  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/c3e3fc86-ec6b-441c-bc22-68ffa6f5e768)  
這個錯誤訊息代表你安裝的檔案cp是不對的，你可以直接去安裝有對應你python的版本的  
又或者你可以直接更改虛擬環境成對應的python版本  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/c865ee70-d5d7-42b7-ae87-71259cac39a4)  

### 9. 測試有沒有調用gpu成功。  
![image](https://github.com/wangbosen123/YOLOv7-/assets/92494937/83da983c-1f9d-48e8-a420-9770328f878a)

### 10. 接者就可以訓練YOLOv7了  
"可能遇到多線程的問題"(訓練模型時如果遇到workers的問題加入以下程式碼在train.py裡面主程式的部分便可以解決):  
解決方式為在主程式前面加上，os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
如果遇到其他沒有安裝的套件就pip install 目標套件  這樣及可以順利安裝  
