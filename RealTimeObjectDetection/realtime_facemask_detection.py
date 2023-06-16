import os, sys

# setup paths
WORKSPACE_PATH = './Tensorflow/workspace'
SCRIPTS_PATH = './Tensorflow/scripts'
APIMODEL_PATH = './Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
# Create label map
labels = [{'name':'Mask', 'id':1}, {'name':'NoMask', 'id':2}]
## label map이 label_map.pbtxt 로 되어있지 않으면 변경하기
'''
with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
'''
# Create TF recodes_tf 레코드를 생성하는 스크립트 실행
'''
/Users/dobby/Documents/arcticfo111/msaischool_23/real_time_facemask_detection/RealTimeObjectDetection/Tensorflow/workspace/images

terminal_command1 ='./Tensorflow/scripts/generate_tfrecord.py' -x './Tensorflow/workspace/images/train' -l './Tensorflow/workspace/annotations/label_map.pbtxt' -o './Tensorflow/workspace/annotations/train.record'
terminal_command2 ='./Tensorflow/scripts//generate_tfrecord.py' -x './Tensorflow/workspace/images/test' -l './Tensorflow/workspace/annotations/label_map.pbtxt' -o './Tensorflow/workspace/annotations/test.record'
os.system(terminal_command1, terminal_command2)
'''
# download TF Models Pretrained Models from Tensorflow Model Zoo
'''

'''