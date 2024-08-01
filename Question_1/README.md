# Áp dụng Multiclass SVM trên tập dữ liệu chữ viết tay MNIST

## 1. SVM
Là một mô hình supervised machine với mục tiêu chính là tìm ra các hyperplane tối ưu nhất để phân tách các điểm dữ liệu thuộc các class khác nhau.

SVM có khả năng xử lý dữ liệu nhiều chiều và được sử dụng rộng rãi trong nhiều ứng dụng như nhận diện hình ảnh, phân loại văn bản,...

Ở bài toán phân biệt loại kí tự quang học này thì mình có mục tiêu là phân loại các ảnh thành 10 class từ 0 tới 9, thế nên mình sẽ dùng Multi-class SVM với chiến thuật One vs All

##2. Mô tả cách giải quyết bài toán
- **Bước 1**: 
    - Nhận vào 2 vector là X(vector ảnh) với y(vector label) train với dimension tương ứng là (786,) và (1000,), ta sẽ có được các class từ 0 tới 9 với np.unique(y)
- **Bước 2**: 
    - Với từng class, ta tiến hành train một SVM binary classifier cho class đó để có thể classify class đó khỏi những class khác, sau đó ta sẽ lưu các classifier này vào chung một list models.
- **Bước 3**:
    -   Khi ta đưa một sample vào Multi-class SVM, thì sample này sẽ được đưa vào từng SVM classifier một trong list models, mỗi model này sẽ output ra một số điểm dự đoán(có 10 class thì ta có một list chứa 10 element), ta dùng np.argmax để output ra được class mà Multi-class SVM đã dự đoán

##3. Mô tả các files
- `main.py`: Chứa một simple pipeline dùng Multi-class SVM trên tập dữ liệu MNIST.
- `model.py`: Triển khai Random Forest scratch sử dụng Numpy.
- `utils.py`: Load MNIST dataset
