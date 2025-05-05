import os
import sys
import subprocess
import shutil
import re
import fnmatch
import tarfile
import tensorflow as tf
import io
from PIL import Image
import glob
import random
import zipfile
import tkinter as tk
from tkinter import filedialog
import threading
import time
from datetime import datetime

# Set up environment and paths
def setup_environment():
    """Initialize environment variables and folder structure"""
    if os.name == 'nt':  # Windows
        HOMEFOLDER = os.getcwd() + "\\"
        FINALOUTPUTFOLDER_DIRNAME = 'final_output'
        FINALOUTPUTFOLDER = os.path.join(HOMEFOLDER, FINALOUTPUTFOLDER_DIRNAME)
        MLENVIRONMENT = "WINDOWS"
    else:  # Linux/Colab
        os.environ["HOMEFOLDER"] = "/content/"
        HOMEFOLDER = os.environ["HOMEFOLDER"]
        FINALOUTPUTFOLDER_DIRNAME = 'final_output'
        FINALOUTPUTFOLDER = HOMEFOLDER + FINALOUTPUTFOLDER_DIRNAME
        MLENVIRONMENT = "COLAB"
    
    # Create output folders
    if os.path.exists(FINALOUTPUTFOLDER) and os.path.isdir(FINALOUTPUTFOLDER):
        shutil.rmtree(FINALOUTPUTFOLDER)
    os.makedirs(FINALOUTPUTFOLDER, exist_ok=True)
    
    # Create base directories
    sources_folder = os.path.join(HOMEFOLDER, "Sources")
    images_folder = os.path.join(sources_folder, "images")
    train_folder = os.path.join(images_folder, "train")
    test_folder = os.path.join(images_folder, "test")
    models_folder = os.path.join(sources_folder, "models")
    
    for folder in [sources_folder, images_folder, train_folder, test_folder, models_folder]:
        os.makedirs(folder, exist_ok=True)
    
    print(f"Environment: {MLENVIRONMENT}")
    print(f"Home folder: {HOMEFOLDER}")
    print(f"Output folder: {FINALOUTPUTFOLDER}")
    
    return HOMEFOLDER, FINALOUTPUTFOLDER, MLENVIRONMENT

def run_command(cmd, shell=True):
    """Execute a shell command and print output"""
    print("Running:", cmd)
    try:
        subprocess.check_call(cmd, shell=shell)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return False

def ask_for_zip():
    """Open file dialog to select TFRecords ZIP file"""
    root = tk.Tk()
    root.withdraw()
    print("📂 Please select your TFRecords ZIP file.")
    zip_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
    if not zip_path:
        print("❌ No file selected. Exiting.")
        sys.exit(1)
    return zip_path

def extract_zip_to_sources(zip_path, extract_path):
    """Extract the ZIP file to the sources directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extracted zip to:", extract_path)

def save_checkpoint_periodically(model_dir, interval=300):
    """Save training checkpoints periodically"""
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{timestamp}")
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        time.sleep(interval)  # Default: every 5 minutes

def find_files(directory, pattern):
    """Find files matching a pattern in a directory"""
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield os.path.join(root, basename)

def set_tfrecord_variables(directory):
    """Find TFRecord and label map files in the directory"""
    train_record_fname = ''
    val_record_fname = ''
    label_map_pbtxt_fname = ''
    
    # Search for TFRecord files
    for tfrecord_file in find_files(directory, '*.tfrecord'):
        if '/train/' in tfrecord_file or 'train' in tfrecord_file.lower():
            train_record_fname = tfrecord_file
        elif '/valid/' in tfrecord_file or 'val' in tfrecord_file.lower():
            val_record_fname = tfrecord_file
    
    # Search for label map files
    for label_map_file in find_files(directory, '*_label_map.pbtxt'):
        label_map_pbtxt_fname = label_map_file
    
    if not label_map_pbtxt_fname:
        for label_map_file in find_files(directory, '*.pbtxt'):
            label_map_pbtxt_fname = label_map_file
            break

    print(f"Found train record: {train_record_fname}")
    print(f"Found validation record: {val_record_fname}")
    print(f"Found label map: {label_map_pbtxt_fname}")
    
    return train_record_fname, val_record_fname, label_map_pbtxt_fname

def setup_tensorflow_models(homefolder):
    """Set up TensorFlow models repository"""
    tmpModelPath = os.path.join(homefolder, 'models')
    if os.path.exists(tmpModelPath) and os.path.isdir(tmpModelPath):
        shutil.rmtree(tmpModelPath)
    
    # Clone TensorFlow models repo
    run_command('git clone --depth 1 https://github.com/tensorflow/models')
    
    # Checkout specific commit that works with our setup
    if os.name == 'nt':
        run_command(f'cd /d {homefolder}models && git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81 && git checkout ad1f7b56943998864db8f5db0706950e93bb7d81')
    else:
        run_command(f'cd {homefolder}models && git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81 && git checkout ad1f7b56943998864db8f5db0706950e93bb7d81')
    
    # Compile protobuf files
    run_command(f'cd {homefolder}models/research && protoc object_detection/protos/*.proto --python_out=.')
    
    # Modify setup file for compatibility
    with open(f'{homefolder}models/research/object_detection/packages/tf2/setup.py') as f:
        s = f.read()
    with open(f'{homefolder}models/research/setup.py', 'w') as f:
        s = re.sub('tf-models-official>=2.5.1', 'tf-models-official==2.15.0', s)
        f.write(s)
    
    # Install TensorFlow models package
    run_command(f'pip install {homefolder}models/research/')
    
    # Install TensorFlow
    if os.name != 'nt':
        run_command('pip install tensorflow==2.15.0')
    else:
        print("Make sure tensorflow==2.15.0 is installed on your system.")
    
    # Test the installation
    run_command(f'python {homefolder}models/research/object_detection/builders/model_builder_tf2_test.py')

def get_num_classes(pbtxt_fname):
    """Get the number of classes from label map file"""
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

def get_classes(pbtxt_fname):
    """Get the class names from label map file"""
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return [category['name'] for category in category_index.values()]

def create_label_file(filename, labels):
    """Create a label file from class names"""
    with open(filename, 'w') as file:
        for label in labels:
            file.write(label + '\n')

def setup_model_config(homefolder, train_record_fname, val_record_fname, label_map_pbtxt_fname, model_type='ssd-mobilenet-v2'):
    """Set up model configuration"""
    # Model configurations
    MODELS_CONFIG = {
        'ssd-mobilenet-v2': {
            'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
            'base_pipeline_file': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
            'pretrained_checkpoint': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        }
    }
    
    # Get model parameters
    model_name = MODELS_CONFIG[model_type]['model_name']
    pretrained_checkpoint = MODELS_CONFIG[model_type]['pretrained_checkpoint']
    base_pipeline_file = MODELS_CONFIG[model_type]['base_pipeline_file']
    
    # Create model directory
    run_command(f'mkdir -p {homefolder}models/mymodel/')
    
    # Download pre-trained model
    download_tar = 'https://downloads.limelightvision.io/models/' + pretrained_checkpoint
    run_command(f'wget {download_tar} -P {homefolder}models/mymodel/')
    
    # Extract pre-trained model
    tar = tarfile.open(f"{homefolder}models/mymodel/{pretrained_checkpoint}")
    tar.extractall(f"{homefolder}models/mymodel/")
    tar.close()
    
    # Download model config
    download_config = 'https://downloads.limelightvision.io/models/' + base_pipeline_file
    run_command(f'wget {download_config} -P {homefolder}models/mymodel/')
    
    # Training parameters
    num_steps = 40000
    checkpoint_every = 2000
    batch_size = 16
    
    # Get paths
    pipeline_fname = f'{homefolder}models/mymodel/{base_pipeline_file}'
    fine_tune_checkpoint = f'{homefolder}models/mymodel/{model_name}/checkpoint/ckpt-0'
    
    # Get class information
    num_classes = get_num_classes(label_map_pbtxt_fname)
    classes = get_classes(label_map_pbtxt_fname)
    print('Total classes:', num_classes)
    print('Classes:', classes)
    
    # Create labels file
    create_label_file(f"{homefolder}limelight_neural_detector_labels.txt", classes)
    
    # Create custom config file
    print('Writing custom configuration file')
    with open(pipeline_fname) as f:
        s = f.read()
    with open(f'{homefolder}pipeline_file.config', 'w') as f:
        s = re.sub('fine_tune_checkpoint: ".*?"', f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
        s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', f'input_path: "{train_record_fname}"', s)
        s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', f'input_path: "{val_record_fname}"', s)
        s = re.sub('label_map_path: ".*?"', f'label_map_path: "{label_map_pbtxt_fname}"', s)
        s = re.sub('batch_size: [0-9]+', f'batch_size: {batch_size}', s)
        s = re.sub('num_steps: [0-9]+', f'num_steps: {num_steps}', s)
        s = re.sub('checkpoint_every_n: [0-9]+', f'num_classes: {num_classes}', s)
        s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "detection"', s)
        if model_type == 'ssd-mobilenet-v2':
            s = re.sub('learning_rate_base: .8', 'learning_rate_base: .004', s)
            s = re.sub('warmup_learning_rate: 0.13333', 'warmup_learning_rate: .0016666', s)
        f.write(s)
    
    return model_type, num_steps, checkpoint_every

def train_model(homefolder, checkpoint_every, num_steps):
    """Train the object detection model"""
    training_dir = f'{homefolder}training_progress'
    
    # Remove existing training progress
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    
    # Start training
    print(f"Starting training with {num_steps} steps, checkpoints every {checkpoint_every} steps")
    run_command(f'python {homefolder}models/research/object_detection/model_main_tf2.py ' + 
                f'--pipeline_config_path={homefolder}pipeline_file.config ' +
                f'--model_dir={training_dir}/ ' +
                f'--alsologtostderr ' +
                f'--checkpoint_every_n={checkpoint_every} ' +
                f'--num_train_steps={num_steps} ' +
                f'--num_workers=2 ' +
                f'--sample_1_of_n_eval_examples=1')
    
    return training_dir

def export_model(homefolder, training_dir, output_dir):
    """Export the trained model"""
    exporter_path = f'{homefolder}models/research/object_detection/export_tflite_graph_tf2.py'
    
    # Export model to TensorFlow SavedModel format
    run_command(f'python {exporter_path} ' +
                f'--trained_checkpoint_dir {training_dir} ' +
                f'--output_directory {output_dir} ' +
                f'--pipeline_config_path {homefolder}pipeline_file.config')
    
    # Convert to TFLite (32-bit)
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{output_dir}/saved_model')
    tflite_model = converter.convert()
    model_path_32bit = f'{output_dir}/limelight_neural_detector_32bit.tflite'
    with open(model_path_32bit, 'wb') as f:
        f.write(tflite_model)
    
    # Copy labels and config
    run_command(f'cp {homefolder}limelight_neural_detector_labels.txt {output_dir}')
    run_command(f'cp {homefolder}pipeline_file.config {output_dir}')
    
    return model_path_32bit

def extract_images_from_tfrecord(tfrecord_path, output_folder, num_samples=100):
    """Extract sample images from TFRecord for quantization"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    saved_images = 0
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in raw_dataset.take(num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        image_data = example.features.feature['image/encoded'].bytes_list.value[0]
        image = Image.open(io.BytesIO(image_data))
        image.save(os.path.join(output_folder, f'image_{saved_images}.png'))
        saved_images += 1
        if saved_images >= num_samples:
            break
    
    print(f"Extracted {saved_images} images to {output_folder}")
    return saved_images > 0

def optimize_model(homefolder, model_path_32bit, train_record_fname, output_dir):
    """Optimize the model with quantization"""
    # Extract sample images from TFRecord
    extracted_sample_folder = f'{homefolder}extracted_samples'
    if os.path.exists(extracted_sample_folder) and os.path.isdir(extracted_sample_folder):
        shutil.rmtree(extracted_sample_folder)
    
    success = extract_images_from_tfrecord(train_record_fname, extracted_sample_folder)
    if not success:
        print("Failed to extract sample images for quantization")
        return
    
    quant_image_list = glob.glob(extracted_sample_folder + '/*')
    print("Using samples from " + extracted_sample_folder)
    print("Found samples: " + str(len(quant_image_list)))
    
    # Get model input dimensions
    interpreter = tf.lite.Interpreter(model_path=model_path_32bit)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    # Define representative data generator for quantization
    def representative_data_gen():
        dataset_list = quant_image_list
        quant_num = min(300, len(dataset_list))
        for i in range(quant_num):
            pick_me = random.choice(dataset_list)
            image = tf.io.read_file(pick_me)
            if pick_me.lower().endswith(('.jpg', '.jpeg')):
                image = tf.io.decode_jpeg(image, channels=3)
            elif pick_me.lower().endswith('.png'):
                image = tf.io.decode_png(image, channels=3)
            elif pick_me.lower().endswith('.bmp'):
                image = tf.io.decode_bmp(image, channels=3)
            image = tf.image.resize(image, [width, height])
            image = tf.cast(image / 255., tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]
    
    # Convert to 8-bit quantized model
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{output_dir}/saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    model_path_8bit = f'{output_dir}/limelight_neural_detector_8bit.tflite'
    with open(model_path_8bit, 'wb') as f:
        f.write(tflite_model)
    
    return model_path_8bit

def compile_for_edgetpu(output_dir, mlenvironment):
    """Compile the model for Edge TPU if on Linux"""
    if mlenvironment == "COLAB":
        # Install Edge TPU compiler
        run_command('curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -')
        run_command('echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list')
        run_command('sudo apt-get update')
        run_command('sudo apt-get install -y edgetpu-compiler')
        
        # Compile for Edge TPU
        run_command(f'cd {output_dir} && edgetpu_compiler limelight_neural_detector_8bit.tflite && mv limelight_neural_detector_8bit_edgetpu.tflite limelight_neural_detector_coral.tflite')
    else:
        print("Edge TPU compilation is only available in Linux/Colab environment")

def create_output_zip(homefolder, output_dir):
    """Create a ZIP archive of the output"""
    zip_path = f'{homefolder}limelight_detectors.zip'
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # Get base directory name
    base_dir = os.path.basename(output_dir)
    parent_dir = os.path.dirname(output_dir)
    
    # Create ZIP file
    shutil.make_archive(zip_path[:-4], 'zip', parent_dir, base_dir)
    print(f"Created ZIP archive at: {zip_path}")

def main():
    """Main function to run the entire process"""
    print("=" * 80)
    print("Custom Object Detection Training Pipeline")
    print("=" * 80)
    
    # Setup environment
    HOMEFOLDER, FINALOUTPUTFOLDER, MLENVIRONMENT = setup_environment()
    
    # Ask for zip file and extract it
    zip_path = ask_for_zip()
    extract_zip_to_sources(zip_path, os.path.join(HOMEFOLDER, "Sources"))
    
    # Find TFRecord files
    train_record_fname, val_record_fname, label_map_pbtxt_fname = set_tfrecord_variables(HOMEFOLDER)
    
    if not train_record_fname or not val_record_fname or not label_map_pbtxt_fname:
        print("❌ Could not find all required files in the ZIP. Make sure it contains:")
        print("   - Training TFRecord (with 'train' in the filename or path)")
        print("   - Validation TFRecord (with 'val' or 'valid' in the filename or path)")
        print("   - Label map file (with extension .pbtxt)")
        sys.exit(1)
    
    # Setup TensorFlow models repository
    setup_tensorflow_models(HOMEFOLDER)
    
    # Setup model configuration
    model_type, num_steps, checkpoint_every = setup_model_config(
        HOMEFOLDER, train_record_fname, val_record_fname, label_map_pbtxt_fname
    )
    
    # Start checkpoint saving thread
    checkpoint_thread = threading.Thread(
        target=save_checkpoint_periodically, 
        args=(os.path.join(HOMEFOLDER, "training_progress"),),
        daemon=True
    )
    checkpoint_thread.start()
    
    # Train the model
    training_dir = train_model(HOMEFOLDER, checkpoint_every, num_steps)
    
    # Export and optimize the model
    model_path_32bit = export_model(HOMEFOLDER, training_dir, FINALOUTPUTFOLDER)
    model_path_8bit = optimize_model(HOMEFOLDER, model_path_32bit, train_record_fname, FINALOUTPUTFOLDER)
    
    # Compile for Edge TPU (if in Linux/Colab)
    compile_for_edgetpu(FINALOUTPUTFOLDER, MLENVIRONMENT)
    
    # Create ZIP archive
    create_output_zip(HOMEFOLDER, FINALOUTPUTFOLDER)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Model files are in: {FINALOUTPUTFOLDER}")
    print(f"ZIP archive is at: {HOMEFOLDER}limelight_detectors.zip")
    print("=" * 80)

if __name__ == "__main__":
    main()