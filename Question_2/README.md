# Triplet Loss
## 1. Định nghĩa
Triplet Loss là một hàm loss thường được sử dụng trong các bài toán Image Regconition và Matching Problems, tiêu biểu là metric learning để nhận diện khuôn mặt. Mục tiêu của triplet los là nhằm học ra được mộ embedding space mà trong đó các items giống nhau thì nằm gần nhau, còn các item khác nhau thì sẽ cách xa nhau

## 2. Công thức Triplet Loss với One Samples (Question 2 - a)
Triplet Loss được biểu diễn:

$$L(a, p, n) = \max\left(0, \| f(a) - f(p) \|^2 - \| f(a) - f(n) \|^2 + \alpha\right)$$

Ở công thức trên:
- $f(a)$ là embedding vector của anchor.
- $f(p)$ là embedding vector của positive sample (cùng class với anchor).
- $f(n)$ là embedding vector của negative sample (khác class với anchor).
- $α$ là margin phân cách cặp positive và negative.

## 3. Công thức Triplet Loss với Multiple Samples (Question 2 - b)
$$L(a, p, n) = \frac{1}{A} \sum_{i=1}^{N} \max\left(0, \frac{1}{P} \sum_{p\in P}\|f(a) - f(p)\|^2 - \frac{1}{N}\sum_{n\in N} \|f(a) - f(n)\|^2 + \alpha\right)$$

Ở công thức trên:
- $f(a)$ là embedding vector của anchor.
- $f(p)$ là embedding vector của positive sample (cùng class với anchor).
- $f(n)$ là embedding vector của negative sample (khác class với anchor).
- $α$ là margin phân cách cặp positive và negative.
- $P$ là số lượng của positive samples.
- $N$ là số lượng của negative samples.
- $A$ là số lượng triplets được tính.

## 4. Giải thích công thức
### 4.1. Hàm Embedding $f$
- $f(x)$: Một hàm (thường là mạng nơ-ron) ánh xạ một đầu vào $x$ đến một embedding space nơi có thể thực hiện các so sánh.
### 4.2. Distance Metric
- $\| f(a) - f(p) \|^2$: Khoảng cách bình phương giữa anchor và positive sample trong không gian embedding. (L2 Distance)

- $\|f(a) - f(n)\|^2 $: Khoảng cách bình phương giữa anchor và negative sample trong không gian embedding. (L2 Distance)
### 4.3. Margin $\alpha $
- Margin được áp dụng giữa các cặp positive và negative. Margin giúp đảm bảo negative samples cách xa the anchor hơn positive examples ít nhất một khoảng $\alpha$.
### 4.4. Loss Calculation
- Nếu khoảng cách của cặp anchor và positive sample không đủ nhỏ hơn khoảng cách của cặp anchor và negative sample ít nhất là $\alpha$, thì biểu thức $\|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha$ sẽ dương và sẽ được tính vào hàm loss.

- Hàm $\max$ đảm bảo rằng chỉ có các giá trị dương mới góp phần vào  hàm loss. Nếu sự khác biệt là âm (tức là bộ ba đã thỏa mãn điều kiện), hàm loss cho bộ ba đó là bằng không.

## 5. Ưu, nhược điểm
### 5.1. Ưu điểm
- Discriminative Feature Learning
- Robust to Class Imbalance
- Good for High-Dimensional Data

### 5.2. Nhược điểm
- Margin Sensitivity
- Selection of Triplets
- Computational Cost
## 6. Ứng dụng
- **Face Recognition**: ảm bảo rằng các khuôn mặt của cùng một người gần nhau hơn trong không gian embedding so với các khuôn mặt của những người khác nhau.
- **Image Retrieval**: Giúp học các nhúng sao cho các hình ảnh tương tự gần nhau hơn trong không gian nhúng.

## 7. Files
- `triplet_loss_one_sample.py`:  Chứa triển khai Triplet Loss với 1 mẫu neo và 1 mẫu giả
- `triplet_loss_multi_samples.py`: Chứa triển khai Triplet Loss với 2 mẫu neo và 5 mẫu giả
- `notebook/Triplet_Loss.ipynb`: Chứa ví dụ triển khai với một mẫu và nhiều mẫu
