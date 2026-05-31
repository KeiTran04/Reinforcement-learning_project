# BÁO CÁO CHI TIẾT DỰ ÁN: ĐIỀU KHIỂN ĐÈN GIAO THÔNG THÔNG MINH ĐA GIAO LỘ SỬ DỤNG HỌC TĂNG CƯỜNG ĐA TÁC TỬ (MARL)

Báo cáo này được cấu trúc theo chuẩn đề tài nghiên cứu/đồ án môn học/đồ án tốt nghiệp đại học, giúp bạn dễ dàng nắm vững kiến thức từ A đến Z để trình bày và bảo vệ trước Hội đồng giảng viên.

---

## PHẦN 1: ĐẶT VẤN ĐỀ & MỤC TIÊU ĐỀ TÀI

### 1. Bối cảnh & Thực trạng
* **Ùn tắc giao thông đô thị** là một trong những bài toán nhức nhối nhất tại các thành phố lớn.
* **Hạn chế của hệ thống đèn truyền thống:** Đèn giao thông hiện tại hầu hết hoạt động theo chu kỳ cố định (Fixed-time Control) hoặc cảm biến đơn giản tại chỗ. Chúng không thể tự động thay đổi nhịp đèn theo lưu lượng xe thực tế và hoàn toàn độc lập, không có sự phối hợp giữa các ngã tư liền kề, dẫn đến hiện tượng "dòng xe vừa thoát ngã tư này lại bị chặn đỏ ở ngã tư tiếp theo".

### 2. Ý tưởng giải quyết
* Áp dụng **Trí tuệ nhân tạo (AI)**, cụ thể là **Học tăng cường đa tác tử (Multi-Agent Reinforcement Learning - MARL)** để biến mỗi ngã tư thành một thực thể thông minh (Agent) tự đưa ra quyết định tối ưu dựa trên camera cảm biến dòng xe.
* Các Agent chia sẻ mô hình học tập (**Parameter Sharing**) và trao đổi thông tin trạng thái với hàng xóm lân cận để phối hợp điều khiển nhịp nhàng toàn bộ mạng lưới giao thông.

### 3. Mục tiêu cụ thể của dự án
1. Xây dựng môi trường mô phỏng mạng lưới giao thông lưới $2 \times 2$ (4 ngã tư) thực tế trên phần mềm **SUMO (Simulation of Urban MOCity)**.
2. Thiết kế và tối ưu hóa hệ thống phần thưởng (**Reward Shaping**) và vector trạng thái (**State Vector**) để AI học được 3 hành vi cốt lõi:
   * **Kéo dài đèn xanh** khi đường đang đông xe.
   * **Rút ngắn/Đổi đèn xanh sớm** khi đường đã vắng xe.
   * **Phối hợp đồng pha** với các ngã tư hàng xóm để tạo làn sóng xanh (Green Wave).
3. Đảm bảo an toàn giao thông thông qua cơ chế **Action Masking** (khống chế thời gian xanh tối thiểu để tránh đèn đổi quá nhanh gây nguy hiểm).

---

## PHẦN 2: KIẾN TRÚC HỆ THỐNG & THIẾT KẾ KỸ THUẬT

Hệ thống được xây dựng trên sự kết hợp giữa **SUMO-GUI** (mô phỏng vật lý xe và đường) và **Python** (xử lý mô hình học máy RL thông qua thư viện PyTorch).

```mermaid
graph TD
    subgraph SUMO Simulation
        Net["Mạng lưới đường (grid2x2.net.xml)"]
        Rou["Luồng xe sinh ngẫu nhiên (grid2x2.rou.xml)"]
        TraCI["TraCI API (Cổng kết nối TCP)"]
    end

    subgraph Python Environment (Gym-like MultiSumoEnv)
        State["Tính toán State Vector (13 đặc trưng)"]
        Reward["Tính toán Reward (Phạt hàng chờ + Thưởng thông luồng + Phối hợp)"]
        ActionMask["Action Masking (Khống chế luật min_green=10s)"]
    end

    subgraph MAPPO Brain (Shared Parameter Model)
        Model["Actor-Critic Neural Network (128 hidden size)"]
        Buffer["BỘ NHỚ KINH NGHIỆM ĐỘC LẬP (A0, A1, B0, B1)"]
    end

    Net --> TraCI
    Rou --> TraCI
    TraCI --> State
    TraCI --> Reward
    State --> ActionMask
    ActionMask --> Model
    Model --> Action["Hành động (0: Giữ, 1: Đổi)"]
    Action --> TraCI
    Reward --> Buffer
    State --> Buffer
```

### 1. Thiết kế Markov Decision Process (MDP) cho Tác tử

Mỗi ngã tư ($A_0, A_1, B_0, B_1$) là một Agent có chung cấu trúc MDP:

#### A. Trạng thái (State Space) - 13 đặc trưng chuẩn hóa
Mỗi bước thời gian, Agent quan sát ngã tư và thu về vector trạng thái gồm:
* `[0..3]`: Hàng chờ chuẩn hóa trên 4 làn đường đi vào ngã tư ($\frac{\text{Số xe dừng}}{\text{Hàng chờ cực đại (15 xe)}}$).
* `[4]`: Pha đèn xanh hiện tại (chuẩn hóa theo số pha).
* `[5]`: Thời gian đã giữ pha đèn xanh hiện tại ($\frac{\text{Giây xanh đã qua}}{30s}$).
* `[6]`: Cờ cho phép đổi đèn (0.0 nếu chưa đủ 10s xanh tối thiểu, 1.0 nếu đã đủ).
* `[7]`: Áp lực ngã tư riêng (Sự chênh lệch giữa số xe chờ làn đỏ và làn xanh).
* `[8]`: Mức sử dụng luồng xanh ($\frac{\text{Số xe đang di chuyển trên làn xanh}}{\text{Dung lượng làn}}$).
* `[9]`: Áp lực trung bình của các ngã tư hàng xóm lân cận.
* `[10]`: Mức độ đồng pha hiện tại với ngã tư hàng xóm (0.0 đến 1.0).
* `[11..12]`: Tọa độ lưới chuẩn hóa (Hàng, Cột) giúp Agent định vị vị trí địa lý của mình.

#### B. Hành động (Action Space)
* **`Action = 0` (giữ):** Tiếp tục duy trì pha đèn hiện tại thêm 5 giây.
* **`Action = 1` (ĐỔI):** Chuyển sang pha đèn tiếp theo. Hệ thống sẽ tự động kích hoạt **3 giây đèn vàng** an toàn trước khi chính thức mở đèn xanh hướng mới.

#### C. Hàm phần thưởng cải tiến (Reward Shaping)
Để định hướng AI học đúng hành vi mong muốn, Reward của mỗi Agent được tính bằng:
$$\text{Reward} = R_{\text{waiting}} + R_{\text{delta}} + R_{\text{green\_bonus}} + R_{\text{switch\_penalty}} + R_{\text{coordination}}$$

| Thành phần | Công thức | Ý nghĩa thực tế |
| :--- | :--- | :--- |
| **1. Phạt thời gian chờ ($R_{\text{waiting}}$)** | $-\frac{\text{Tổng thời gian chờ của xe}}{30.0}$ | Phạt khi có xe bị dừng chờ lâu ở làn đỏ. Hệ số 30 giúp cân bằng độ lớn với các phần thưởng khác. |
| **2. Cải thiện hàng chờ ($R_{\text{delta}}$)** | Clip $\left(\frac{\text{Chờ trước} - \text{Chờ hiện tại}}{30.0}, -2.0, 2.0\right)$ | Thưởng điểm cộng khi hàng chờ giảm đi so với bước trước (khuyến khích giải tỏa ùn tắc). |
| **3. Thưởng luồng xanh ($R_{\text{green\_bonus}}$)** | $\min(\text{Xe đang di chuyển trên làn xanh} \times 0.6, 3.0)$ | Thưởng điểm khi làn xanh đang hoạt động hiệu quả (nhiều xe đang chạy thông thoáng). |
| **4. Phạt đổi đèn thích ứng ($R_{\text{switch\_penalty}}$)** | $\begin{cases} -8.0 & \text{nếu } \ge 4 \text{ xe đang chạy}\\ -4.0 & \text{nếu } \ge 2 \text{ xe đang chạy}\\ -1.5 & \text{nếu } \ge 1 \text{ xe đang chạy}\\ 0.0 & \text{nếu } 0 \text{ xe đang chạy} \end{cases}$ | **Hình phạt cốt lõi:** Phạt cực nặng nếu Agent đổi đèn khi làn xanh vẫn đông xe (bắt buộc kéo dài xanh). Không phạt nếu làn xanh đã hoàn toàn hết xe (khuyến khích đổi sớm để cứu làn đỏ). |
| **5. Thưởng phối hợp ($R_{\text{coordination}}$)** | $+0.3$ với mỗi ngã tư hàng xóm đồng pha | Khuyến khích các ngã tư phối hợp pha nhịp nhàng để tạo điều kiện cho dòng xe chạy liên tục qua nhiều nút giao. |

---

## PHẦN 3: THUẬT TOÁN HUẤN LUYỆN & CƠ CHẾ AN TOÀN

### 1. Thuật toán Multi-Agent PPO (MAPPO) với Parameter Sharing
* **Parameter Sharing (Chia sẻ tham số):** Thay vì huấn luyện 4 mạng neural độc lập cho 4 ngã tư, dự án sử dụng **1 mạng Actor-Critic chung** duy nhất. 
  * Cả 4 Agent đều đẩy trạng thái của mình vào mạng chung này để lấy hành động ra.
  * Toàn bộ kinh nghiệm từ 4 ngã tư đều được lưu vào chung một bộ dữ liệu huấn luyện để cập nhật trọng số cho mạng neural.
  * **Ưu điểm:** Giảm dung lượng mô hình, tăng tốc độ hội tụ gấp 4 lần, và giúp các ngã tư cư xử đồng bộ theo cùng một bộ quy tắc thông minh.
* **Độc lập bộ nhớ (GAE Advantage):** Mặc dù dùng chung mạng neural, việc tính toán Lợi thế (Advantage) và Lợi nhuận (Returns) vẫn được tính toán riêng biệt cho từng ngã tư dựa trên chuỗi phần thưởng thực tế của ngã tư đó. Điều này đảm bảo tính đúng đắn về mặt toán học của thuật toán Policy Gradient.

### 2. Cơ chế an toàn Action Masking
* **Lý do cần thiết:** Nếu không khống chế, trong giai đoạn đầu khám phá ngẫu nhiên, AI sẽ liên tục đổi đèn (tần suất 1-2 giây/lần), làm hỏng luồng giao thông và gây sụp đổ quá trình học (Policy Collapse).
* **Cách thực hiện:** Khi thời gian xanh chưa đạt tối thiểu (`time_since_switch < 10s`), môi trường sẽ can thiệp và gán xác suất của hành động 1 (ĐỔI) bằng một số âm cực lớn ($-10^9$). Phép tính Softmax sau đó sẽ ép xác suất chọn hành động ĐỔI về đúng `0.00`, ép buộc AI chỉ được chọn hành động GIỮ.

---

## PHẦN 4: KẾT QUẢ THỰC NGHIỆM (KÝ SỰ A ĐẾN Z)

Quá trình phát triển dự án trải qua các cột mốc quan trọng sau:

```mermaid
chronology
    Giai đoạn 1 - Nghiên cứu hệ thống đơn : SUMO, TraCI, Single PPO. Thiết lập nền tảng mô phỏng.
    Giai đoạn 2 - Phát triển đa giao lộ : Lưới 2x2, Parameter Sharing MAPPO.
    Giai đoạn 3 - Gặp lỗi sụp đổ chính sách : Không có Action Masking, đèn đổi liên tục. Reward sụp đổ ở mức -1000.
    Giai đoạn 4 - Sửa lỗi & Bị rập khuôn chu kỳ : Thêm Action Masking giúp ổn định ở mức -326. Tuy nhiên AI bị rập khuôn luôn đổi ở đúng 12s do phạt chờ quá lớn.
    Giai đoạn 5 - Tối ưu hóa & Đạt thích ứng thực tế : Thay đổi tỷ lệ Reward và Entropy. AI đạt điểm dương +129. Thời gian xanh dao động thông minh từ 12s đến 57s.
```

### So sánh Hiệu năng giữa 2 phiên bản huấn luyện:

| Chỉ số đánh giá | Phiên bản cũ (Chưa tối ưu phần thưởng) | Phiên bản mới (Đã tối ưu thích ứng) |
| :--- | :--- | :--- |
| **Đồ thị hội tụ** | Hội tụ phẳng lỳ rất nhanh ở mức âm **`-326`** | Đi lên từng bước vững chắc và ổn định ở mức dương **`+129`** |
| **Hành vi nhịp đèn** | Bị rập khuôn (luôn luôn đổi đèn ở đúng **12 giây**) | Linh hoạt, thích ứng thời gian xanh dao động từ **`12 giây` đến `57 giây`** |
| **Hiệu quả thông xe** | Đổi đèn liên tục kể cả khi đang đông xe (`88` lần đổi) | Đợi xe đi hết mới đổi. Giảm số lần đổi thừa xuống **`67`** lần |
| **Tổng Reward hệ thống** | **`-230.5`** (Tất cả ngã tư đều chịu điểm âm) | **`+137.1`** (Tất cả ngã tư đều đạt điểm dương) |

---

## PHẦN 5: CẨM NANG BẢO VỆ ĐỒ ÁN TRƯỚC GIẢNG VIÊN

Khi trình bày đề tài này trước hội đồng giảng viên, bạn nên tập trung làm nổi bật các từ khóa chuyên ngành và chuẩn bị sẵn câu trả lời cho các câu hỏi thường gặp sau:

### Slide thuyết trình cốt lõi cần có:
1. **Slide Đặt vấn đề:** Chỉ ra sự yếu kém của đèn cố định truyền thống và sự tắc nghẽn giao thông.
2. **Slide Kiến trúc:** Sơ đồ kết nối Python - TraCI - SUMO.
3. **Slide Thuật toán:** Giải thích cách hoạt động của MAPPO và cơ chế Parameter Sharing (Tại sao 4 ngã tư dùng chung 1 bộ não).
4. **Slide MDP:** Trình bày chi tiết Vector trạng thái (State) và đặc biệt là cách thiết kế Hàm phần thưởng (Reward Shaping) - đây là phần giảng viên đánh giá cao nhất vì thể hiện tư duy thiết kế hệ thống.
5. **Slide Kết quả:** So sánh trực quan biểu đồ hội tụ cũ (bị rập khuôn) và mới (thích ứng động) cùng kết quả số lần đổi đèn và tổng reward.

### Các câu hỏi phản biện thường gặp của Giảng viên & Gợi ý trả lời:

#### Câu 1: Tại sao em lại sử dụng cơ chế "Parameter Sharing" thay vì huấn luyện các tác tử độc lập?
> **Gợi ý trả lời:** 
> *Thưa thầy/cô, việc sử dụng chung tham số (Parameter Sharing) mang lại ba lợi ích lớn:*
> 1. *Giảm đáng kể chi phí tính toán và số lượng tham số mạng neural cần cập nhật, giúp hệ thống chạy nhanh hơn.*
> 2. *Tăng tính ổn định và tốc độ hội tụ vì mô hình được học từ lượng dữ liệu phong phú gấp 4 lần từ cả 4 ngã tư đồng thời.*
> 3. *Tạo ra sự đồng nhất trong hành vi điều khiển. Tất cả các ngã tư sẽ hành xử theo cùng một hệ quy tắc thông minh tương tự nhau.*

#### Câu 2: Làm thế nào để các ngã tư phối hợp với nhau để tạo "sóng xanh" mà không cần một bộ điều khiển trung tâm?
> **Gợi ý trả lời:**
> *Hệ thống của em là hệ thống phi tập trung (Decentralized). Sự phối hợp được thực hiện gián tiếp thông qua việc thiết kế State và Reward:*
> 1. *Trong State của mỗi tác tử, em đưa thông tin về **Áp lực trung bình của hàng xóm** và **Độ lệch pha với hàng xóm**.*
> 2. *Trong Reward, em thiết lập một điểm cộng **Coordination Bonus (+0.3)** khi tác tử chuyển đèn đồng pha hoặc phối hợp tốt với ngã tư lân cận.*
> *Thông qua việc tối ưu hóa phần thưởng này, các Agent tự động học được cách phối hợp tự phát để tối đa hóa điểm số của toàn mạng lưới.*

#### Câu 3: Vai trò thực tế của cơ chế "Action Masking" ở đây là gì? Tại sao không để AI tự học luật an toàn?
> **Gợi ý trả lời:**
> *Nếu để AI tự học luật an toàn (tự học rằng không nên đổi đèn quá nhanh), AI sẽ mất hàng trăm episodes chỉ để học luật cơ bản này thông qua thử và sai, thậm chí có thể không bao giờ hội tụ do không gian khám phá quá rộng và bất ổn định. Cơ chế Action Masking hoạt động như một bộ lọc cứng (hard constraint) can thiệp trực tiếp vào đầu ra logits của mạng neural. Nó ngăn chặn tuyệt đối việc sinh hành động sai luật (đổi đèn dưới 10s), giúp bảo vệ an toàn vật lý của hệ thống giao thông trong thực tế và giúp AI tập trung hoàn toàn vào việc học cách tối ưu hóa luồng xe khi đã đủ điều kiện an toàn.*
