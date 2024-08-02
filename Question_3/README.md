# FCNN vận dụng Triplet Loss trên tập dữ liệu MNIST

## Mô tả quá trình
- **Ý tưởng**:
    - Dùng mô hình FCNN để học các trích xuất các đặc trưng
    - Dùng Triplet loss để gom cụm các sample có cùng class lại gần nhau, đồng thời cách xa các sample khác class 
- **Training**:
    - Training theo các batch, cứ trong một batch ta lấy ra 3 list từ tập train là anchor, positive, negative với các element trong anchor và positive có cùng class còn các element trong negative sẽ khác class với anchor
    - Với mỗi list ta sẽ forward qua neural network trích xuất được 3 đặc trưng 
    - Đem 3 đặc trưng của anchor, positive, negative để tìm gradient của hàm triplet loss
    - đưa gradients for the loss để chạy backward rồi update các parameter.
- **Inference**: 
    - Sau khi có mô hình được huấn luyện, từ Input (1, 784) qua mô hình MLP (Triplet Loss) sẽ cho ra được đặc trưng Output (1, 64)
    - Ứng với mỗi class, lấy ra 1 sample, rồi đưa qua mô hình FCNN để trích xuất đặc trưng. Sau đó, gộp lại thành một Anchor List
    - Dùng hàm Similarity để so sánh giữa Output và Anchor List, lấy ra Index có Similarity Score cao nhất làm predict class

## Phân tích Ưu & nhược điểm
### Ưu điểm
- **Similarity Learning**: **Distance Metric Learning** Triplet loss là thuật toán tối ưu cho 
similarity metric. Điều này khiến nó trở nên rất hữu dụng trong các task mà mục tiêu là tính toán sự giống nhau hay khoảng cách giữa các sample.

- **Ability to Capture Complex Patterns**: **FCNN** có thể học và mô hình hóa các mối quan hệ phi tuyến tính phức tạp trong dữ liệu nhờ cấu trúc nhiều lớp của nó. Trong khi đó, **SVM** có thể một phần nào đó đạt được hiệu quả nhưng đòi hỏi sự lựa chọn đúng giữa các kernel và tunning parameter, và đây là một quá trình phức tạp và tốn thời .

- **Scalability**: **FCNN** có thể mở rộng với nhiều lớp và nơ-ron hơn để xử lý các tập dữ liệu lớn và phức tạp hơn để có thể cải thiện hiệu suất, còn **SVM** có thể sẽ tốn rất nhiều tài nguyên và thời gian với những dataset  hơn.

### Nhược điểm
- **Overfitting**: **FCNN** dễ bị overfitting hơn, đặc biệt là khi có ít data, còn SVM thì có thể có peformance tốt với lượng data ít
- **Triplet Loss Challenges**: Triển khai và tunning triplet loss và đòi hỏi phải chọn các triplet sao cho tối ưu hàm loss 
- **Computational Resources**: **FCNN** thường sẽ cần nhiều tài nguyên tính toán hơn so với SVM(cụ thể là SVM với linear kernels)

## Các files
- `model.py`: Triển khai FCNN.
- `config.py`: Định nghĩa một số biến learning rate, batch size, ...
- `utils.py`: Chứa một số hàm cần thiết để handle Triplet loss, load_data.
- `main.py`: Chứa một pipeline đơn giản sử dụng FCNN trên tập dữ liệu MNIST.