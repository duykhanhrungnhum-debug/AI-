# Media Generation V2

## Mục tiêu

Giữ production path ngắn, dễ kiểm soát và bám lệnh tự nhiên. Không dùng nhiều planner/router/detector/critic nối tiếp nhau trong lúc người dùng chờ ảnh.

## Ảnh: production path duy nhất

`Natural request -> Qwen3-1.7B recaption -> FLUX.2 Klein 4B -> PNG`

- Recaption chỉ làm một việc: chuyển lệnh tự nhiên sang một mô tả tiếng Anh trung thành, giữ nguyên chủ thể, số lượng, style, framing và các điều cấm.
- Không JSON planner, không subject/style class, không danh sách loài, không anatomy contract.
- FLUX.2 Klein 4B là model ảnh production duy nhất. Một model cho photo, 3D, illustration và hướng tới edit/reference để tránh nhiều engine/fallback.
- Mặc định một ảnh cho mỗi lệnh. Không best-of-N/voting.
- Không visual QA/VLM/detector trong request path. Chỉ kiểm tra kỹ thuật: worker thành công, PNG hợp lệ, hash đúng.

## Chất lượng

Quality gate được chuyển sang regression/certification trước khi nâng model hoặc prompt contract. Bộ regression phải đa dạng: ảnh thật, 3D mascot, người, vật thể, phong cảnh, illustration, số lượng chủ thể và prompt đa ngôn ngữ. Kết quả regression phải được xem trực tiếp trước khi phiên bản được coi là VERIFIED.

Nếu một model/version không đạt regression, sửa hoặc thay model tại bước certification; không thêm detector hoặc heuristic riêng vào production.

## Video

Thiết kế V2 theo hướng factorized đơn giản:

`Natural request -> recaption -> [optional FLUX reference/keyframe] -> video generator`

- Text-only khi không cần khóa nhân vật.
- Khi cần nhất quán nhân vật/cảnh: tạo một keyframe/reference bằng FLUX rồi animate bằng image-to-video, thay vì dựng nhiều tầng model/QA.
- Không visual QA loop trong request path. Chất lượng video cũng được regression/certification offline.

Đây là cùng nguyên tắc với hướng factorized image-conditioned video: tách tạo hình chủ đạo và chuyển động để tăng tính nhất quán mà không tạo deep cascade.

## Bài học bắt buộc từ V1

Không tái đưa vào production các cơ chế đã gây lỗi/lặp: planner JSON schema, semantic class router, planner-generated anatomy counts, GroundingDINO hard gate, branding VLM gate, 2B/8B stacked QA, best-of-N selection, hoặc nhiều image engine fallback.

## Giới hạn thực tế

Kiến trúc V2 được tối ưu cho phần cứng hiện có và độ đơn giản. Chất lượng/tốc độ phải đo bằng benchmark thật; không được tuyên bố ngang Gemini/OpenAI/Meta nếu chưa có bằng chứng benchmark tương đương. Đặc biệt video frontier hiện bị giới hạn mạnh bởi model và GPU, không thể giải quyết chỉ bằng thêm code.
