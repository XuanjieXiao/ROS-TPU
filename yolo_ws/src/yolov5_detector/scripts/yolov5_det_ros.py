#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import ast
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

class YOLOv5:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        if 'use_cpu_opt' in getattr(args, '__dict__', {}):
            self.use_cpu_opt = args.use_cpu_opt
        else:
            self.use_cpu_opt = False
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        
        if self.use_cpu_opt:
            self.handle = sail.Handle(args.dev_id)
            self.output_shapes = []
            for output_name in self.output_names:
                output_shape = self.net.get_output_shape(self.graph_name, output_name)
                self.output_shapes.append(output_shape)
        else:
            self.postprocess = PostProcess(
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
                agnostic=self.agnostic,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
            
    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        
        if self.use_cpu_opt:
            out = {}
            for name in outputs.keys():
                # outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num] * self.output_scales[name]
                out[name] = sail.Tensor(self.handle, outputs[name])
        else:
            # resort
            out_keys = list(outputs.keys())
            ord = []
            for n in self.output_names:
                for i, k in enumerate(out_keys):
                    if n == k:
                        ord.append(i)
                        break
            out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
            
        start_time = time.time()
        outputs = self.predict(input_img, img_num)
        self.inference_time += time.time() - start_time
        
        start_time = time.time()
        if self.use_cpu_opt:
            self.cpu_opt_process = sail.algo_yolov5_post_cpu_opt(self.output_shapes, self.net_w, self.net_h)
            results = self.cpu_opt_process.process(outputs, ori_w_list, ori_h_list, [self.conf_thresh]*self.batch_size, [self.nms_thresh]*self.batch_size, True, self.multi_label)
            results = [np.array(result) for result in results]
        else:
            results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx],conf_scores[idx], x1, y1, x2, y2))
        if conf_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        
    return image
   
def prepare_model(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    
    
    # initialize net
    yolov5 = YOLOv5(args)
    
    yolov5.init()
    return yolov5


def detect_ros(args, yolov5, ):
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
    
    batch_size = yolov5.batch_size
    decode_time = 0.0
    
    
    
    
    
    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                decode_time += time.time() - start_time
                
                img_list.append(src_img)
                filename_list.append(filename)
                # import pdb;pdb.set_trace();
                print('test:01: ', len(img_list), batch_size)
                if (len(img_list) == batch_size or cn == len(filenames)) and len(img_list):
                    print('test:02: ', len(img_list), batch_size)
                    # predict
                    results = yolov5(img_list)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        if args.use_cpu_opt:
                            if len(det.shape) > 1:
                                res_img = draw_numpy(img_list[i], det[:,:4], masks=None, classes_ids=det[:, -2], conf_scores=det[:, -1])
                            else:
                                res_img = img_list[i]
                        else:
                            res_img = draw_numpy(img_list[i], det[:,:4], masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            if args.use_cpu_opt:
                                x1, y1, x2, y2, category_id, score = det[idx]
                            else:
                                x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='./datas/datasets/coco128', help='path of input')
    parser.add_argument('--bmodel', type=str, default='./datas/models/BM1684/yolov5s_v6.1_3output_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.6, help='nms threshold')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')