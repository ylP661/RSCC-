from predict import Change_Perception

if __name__ == '__main__':
    cp = Change_Perception()
    imgA_path = r'path/to/image_A.png'
    imgB_path = r'path/to/image_B.png'
    caption = cp.generate_change_caption(imgA_path, imgB_path)
    print(caption)
