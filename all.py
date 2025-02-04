import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps

def process_image(image_path, output_path,
                  frame_color=(0, 0, 0), frame_width=10,
                  epsilon=0.01,
                  overlay_color=(0, 0, 0), overlay_opacity=128):
    """
    이미지에 프레임 추가, 적대적 노이즈 적용, 반투명한 레이어 적용을 순차적으로 수행합니다.

    :param image_path: 원본 이미지 경로
    :param output_path: 결과 이미지를 저장할 경로
    :param frame_color: 프레임의 색상 (RGB 튜플)
    :param frame_width: 프레임의 두께 (픽셀 단위)
    :param epsilon: 적대적 노이즈의 강도
    :param overlay_color: 반투명 레이어의 색상 (RGB 튜플)
    :param overlay_opacity: 반투명 레이어의 불투명도 (0~255 사이의 값)
    """
    # 1. 이미지 로드 및 RGBA로 변환
    image = Image.open(image_path).convert('RGBA')

    # 2. 프레임 추가
    image_with_frame = ImageOps.expand(image, border=frame_width, fill=frame_color)

    # 3. 적대적 노이즈 적용 (이미지를 RGB로 변환하여 처리)
    image_rgb = image_with_frame.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_rgb).unsqueeze(0)  # 배치 차원 추가

    # 임의의 모델 정의
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.fc1 = torch.nn.Linear(16 * image_tensor.shape[2] * image_tensor.shape[3], 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(-1, 16 * image_tensor.shape[2] * image_tensor.shape[3])
            x = self.fc1(x)
            return x

    model = SimpleModel()
    model.eval()

    # 적대적 노이즈 생성
    image_tensor.requires_grad = True
    output = model(image_tensor)
    target = torch.tensor([0])  # 임의의 타겟 레이블
    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
    model.zero_grad()
    loss.backward()
    sign_data_grad = image_tensor.grad.data.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 텐서를 이미지로 변환
    perturbed_image = perturbed_image.squeeze().detach()
    unloader = transforms.ToPILImage()
    perturbed_image = unloader(perturbed_image)

    # 4. 반투명한 레이어 적용 (RGBA로 변환)
    perturbed_image = perturbed_image.convert('RGBA')
    overlay_layer = Image.new('RGBA', perturbed_image.size, overlay_color + (overlay_opacity,))
    final_image = Image.alpha_composite(perturbed_image, overlay_layer)

    # 결과 이미지 저장 (PNG 형식)
    final_image.save(output_path, 'PNG')
    print(f"처리가 완료된 이미지가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    image_path = 'jun.png'        # 입력 이미지 파일명
    output_path = 'fake.png'      # 출력 이미지 파일명

    # 프레임 설정
    frame_color = (0, 0, 0)               # 검정색 프레임
    frame_width = 10                      # 프레임 두께

    # 적대적 노이즈 설정
    epsilon = 0.02                        # 노이즈 강도

    # 반투명 레이어 설정
    overlay_color = (0, 0, 0)             # 검정색 레이어
    overlay_opacity = 10                 # 레이어 불투명도

    process_image(image_path, output_path,
                  frame_color, frame_width,
                  epsilon,
                  overlay_color, overlay_opacity)