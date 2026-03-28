import os
import cv2
import albumentations as A


class DataAugmentation:

    def __init__(self):
        print("[INFO] DataAugmentation initialised")

    def get_pipeline(self):
        """
        cell image specific augmentation pipeline
        avoids transforms that destroy biological texture (e.g. heavy shear)
        """
        pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.6),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.3),

            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.5
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4
            ),

            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=0.2
            ),

            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        ])
        return pipeline

    def augment_image(self, image_path, n_augmented=5):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Could not read: {image_path}")
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # converts BGR to RGB
        pipeline = self.get_pipeline()
        return [pipeline(image=img)["image"] for _ in range(n_augmented)]   # returns a list of augmented images

    def augment_folder(self, input_dir, output_dir, n_augmented=5):
        """
        Copies originals + saves augmented images into output_dir
        output_dir mirrors the class-folder structure of input_dir
        """
        os.makedirs(output_dir, exist_ok=True)

        for class_name in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            output_class_path = os.path.join(output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".bmp", ".jpg", ".png"))
            ]

            print(f"  Augmenting {class_name}: {len(files)} originals -> "
                  f"{len(files) * (n_augmented + 1)} total")

            for file in files:
                file_path = os.path.join(class_path, file)
                base_name  = os.path.splitext(file)[0]

                # Copy original
                orig_img = cv2.imread(file_path)
                if orig_img is None:
                    continue
                cv2.imwrite(
                    os.path.join(output_class_path, f"{base_name}_orig.jpg"),
                    orig_img
                )

                # Save augmented copies
                aug_images = self.augment_image(file_path, n_augmented=n_augmented)
                for i, aug_img in enumerate(aug_images):
                    save_path = os.path.join(output_class_path, f"{base_name}_aug{i}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        print("\n[INFO] Augmentation complete!")
        print(f"[INFO] Saved to: {output_dir}")


if __name__ == "__main__":
    da = DataAugmentation()    # creates a DataAugmentation object, takes all images in train folder & generates 5 aug imgs per original

    input_folder  = r"F:\Cervival_Cell_CNN\SIPakMed_split_dataset\train"
    output_folder = r"F:\Cervival_Cell_CNN\SIPakMed_augmented\train"

    da.augment_folder(input_folder, output_folder, n_augmented=5)