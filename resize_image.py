import os
from PIL import Image, ImageOps
import argparse

def center_crop_images(input_folder, output_folder, target_size=(256, 256)):
    """
    将图片按比例缩放后，从中心裁剪至目标尺寸
    
    参数:
        input_folder: 输入图片文件夹路径
        output_folder: 输出图片文件夹路径
        target_size: 目标尺寸 (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(input_path) as img:
                # 计算缩放比例，使短边匹配目标尺寸
                width, height = img.size
                target_width, target_height = target_size
                
                # 计算缩放比例（保持宽高比）
                ratio = max(target_width / width, target_height / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 计算裁剪区域（居中）
                left = (img.width - target_width) / 2
                top = (img.height - target_height) / 2
                right = (img.width + target_width) / 2
                bottom = (img.height + target_height) / 2
                
                # 执行裁剪
                img = img.crop((left, top, right, bottom))
                
                # 保存图片
                img.save(output_path)
                print(f"已处理: {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='中心裁剪图片至固定尺寸')
    parser.add_argument('--input', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--width', type=int, default=256, help='目标宽度')
    parser.add_argument('--height', type=int, default=256, help='目标高度')
    
    args = parser.parse_args()
    center_crop_images(args.input, args.output, (args.width, args.height))