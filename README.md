# The Third Eye — Light for every life
_A Voice-Guided AI Navigation System for Blind & Low-Vision Shoppers_
---
_Members: Yiru Li,Xin Meng,Shuaiwei Yu,Haoliang Huang_
## 🏗️ System Overview  
The Third Eye works as an **end-to-end multimodal pipeline** connecting:

- Speech understanding  
- Vision perception  
- Depth / 3D reasoning  
- Navigation instructions  
- Multi-agent orchestration  

Everything runs automatically as the user talks and moves.

---

## 🔧 How It Works (Pipeline)

### **1. Voice Command → Intent Understanding**
User speaks: “Where is the milk?”

Our **STT agent** converts audio → text.  
Then the **Intent agent** extracts the object keyword (e.g., “milk”).

### **2. Real-Time Visual Perception**
The camera continuously captures frames.  
The **VLM agent** analyzes them to:

- detect the target item  
- determine confidence  
- return bounding box and location  

### **3. Multi-Agent Decision System**
If object is found → trigger navigation  
If not → stay silent and wait for new frames  
(avoiding noisy or disruptive feedback)

### **4. 3D Spatial Localization**
Once detected, the 3D reconstruction module estimates:

- the user’s camera pose  
- the object’s approximate 3D coordinates  

This answers:
**Left or right? How far? What angle?**

### **5. Relative Position Calculation**
We compute:

- direction (e.g., “30° right”)  
- distance (e.g., “2.1 m away”)  
- forward / left / right orientation  

### **6. Audio Navigation Output**
Finally, the **Navigation agent** generates natural, clear instructions:

- “The apples are two meters ahead on your right.”  
- “Move slightly left.”  
- “Reach forward


