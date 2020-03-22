# 다중 스케일 위성 영상을 활용한 Faster R-CNN기반 선박 검출 알고리즘



## 요약

데이콘에서 진행한 [[국방] 위성 이미지 객체 검출 대회](https://dacon.io/competitions/official/235492/overview/)에서 mAP 0.745로 최종순위 5위를 기록한 펭수high팀의 접근방법을 소개한다. 대회에서 제공된 위성영상 데이터셋은 3000x3000 해상도의 큰 이미지 크기를 갖으며, 클래스간 불균형,객체 크기 분산이 큰 특징을 갖는다. 본 팀은 클래스간 불균형, 객체 크기의 분산이 큰 문제를 해결하기 위해서 제공된 위성영상 데이터셋을 다중 스케일로 분할하으며 detectron2 프레임워크의 Faster R-CNN을 이용하여 학습하였다. 또한 클래스간 불균형 문제를 해결하기 위해서 극단적으로 클래스 수가 적은 "항공모함" 클래스를 별도의 모델로 분할하여 학습하여 mAP 0.745라는 최종성적을 거두었다. 본 팀이 제안한 알고리즘은 테스트 데이터셋의 Public Score와 Private Score간 mAP 차이가 0.0231으로 다른 상위 6개팀의 mAP 차이인 0.03~0.07보다 눈에 띌 정도로 적어 일반화 성능이 뛰어난 것을 확인하였다.



## 서론

딥러닝 기반의 객체검출 알고리즘은 크게 1-Stage, 2-Stage 알고리즘으로 구분된다. 1-Stage 객체 검출 알고리즘은 OverFeat으로 시작하여 YOLO, YOLO9000, YOLOv3 등으로 이어지며 객체 여부와 박스의 크기를 예측하는 Proposal Network와 박스 내부의 클래스를 분류하는 분류 네트워크가 통합된 구조를 갖기 때문에 2-Stage모델보다 성능은 떨어지나 빠른 추론속도가 장점이다. 2-Stage 객체 검출 알고리즘은 R-CNN으로 시작하여 Fast R-CNN, Faster R-CNN, Mask R-CNN으로 1-Stage 모델과 비교했을 때, Proposal Network와 박스 내부의 클래스를 분류하는 분류 네크워크가 분리되어있는 복잡한 구조로 인해 느린 연산속도가 단점이지만 높은 성능이 장점이다.

공공 객체 탐지 데이터셋으로 알려진 COCO, PASCAL VOC 데이터셋은 클래스 불균형 문제를 항상 내포하고있으며, 객체 탐지를 위한 이미지 영역은 대부분 Background로 채워져있다는 추가적인 불균형 문제도 있다. 이로 인해 특정 클래스의 성능이 떨어지며 이를 막기위해 RetinaNet은 Background class와 같이 클래스 수가 많아 손실 함수에 빠르게 수렴되어 작은 손실값을 갖는 예를 easy example, 클래스 수가 적어 큰 손실값을 갖는 예를 hard example로 정의하여 손실함수의 비중을 다르게 하는 방법을 제안하였다. MegDet은 분류 모델과 다르게 큰 이미지 해상도로 인해 객체탐지 모델의 학습 배치가 적은 것이 객체탐지 모델에서 사용하는 BatchNorm 통계값에 악영향을 미쳐 성능하락이 발생한다는 점을 지적하며 128개의 GPU를 사용하여 학습 배치를 크게 적용하는 방법을 제안하였다.

딥러닝 기반의 객체검출 알고리즘은 모두 다중 스케일에서의 강건성이 약한 것으로 알려져있으며 특히 작은 객체 탐지율이 떨어지는 것으로 알려져있다. 이는 컨볼루션 네트워크와 풀링레이어에서 특징맵의 해상도 소실로 인해 작은 객체의 특징이 소실되는 것이 주 원인으로 알려져있으며 이를 보완하기 위해서 뉴럴네트워크의 중간 레이어의 다양한 크기의 특징맵에서 객체탐지 후에 이를 합치는 FPN, 재귀적으로 특징맵을 합치는 RRC(recurrent rolling convolution), 다양한 크기의 특징맵을 병합할 때 가중치를 주는 BiFPN을 제안한 EfficientDet이 있다.

SNIP과 SNIPER는 전이 학습을 위한 학습데이터로 사용되는 ImageNet데이터와 객체 탐지 데이터셋의 스케일 분포가 서로 상이한 것을 지적하며 객체 탐지 모델을 학습할 때, 스케일 분포를 일관성있게 만들어 학습하는 방법을 제안했다. AutoFocus는 SNIPER가 다중 스케일 추론을하여 추론시간이 증가하는 단점을 보완하기 위해 Focus Chip Generation이라는 별도의 브랜치 네트워크 구성을 제안하였다.

SCRDet은 작은 객체를 잘 탐지하기 위해서 장애가되는 "불충분한 객체 특징 정보"와 "부적절한 앵커 샘플링"을 해소하기 위해 최대로 객체 박스와 중첩될 수 있는 EMO 점수를 예측할 수 있으면서 FPN에서 나오는 특징맵을 잘 혼합할 수 있는 SF-Net을 제안하였으며 다중 스케일에서 학습할 수 있는 어텐션 네트워크인 MDA-Net을 제안하였다.

본 팀은 4개월이라는 짧은 기간 동안 최고의 성능을 달성하기 위해서 앞서 설명한 관련 논문들에 착안하여 1). "Box Scale, Rotation Angle, Aspect Ratio를 분석을 통한  앵커 최적화", 2) "다중스케일에서 객체 검출 모델 학습", 3). "클래스 불균형 문제를 해소하기 위한 모델 분리"로 접근하였으며 최종적으로 mAP 0.745라는 최종성적을 거두었다. 본 팀이 제안한 알고리즘은 테스트 데이터셋의 Public Score와 Private Score간 mAP 차이가 0.0231으로 다른 상위 6개팀의 mAP 차이인 0.03~0.07보다 눈에 띌 정도로 적어 일반화 성능이 뛰어나다는 것을 확인하였다.



## 방법론

### 데이터 분석

- Martin 작성



### Detectron2 with Rotated bbox

#### 개요
[Detectron2](https://github.com/facebookresearch/detectron2)는 Facebook AI Research조직에서 만든 오픈소스 프로젝트로 detection을 포함한 SOTA 알고리즘을 구현한 프로젝트이다. 64명의 direct contributor와 약 27,000명의 contributor로 구성되어 있는만큼 매우 매력적인 프로젝트라고 할 수 있다.

Detectron2를 사용하면서 기대효과는 다음과 같다.

- 위성영상 인식 문제를 해결함에 있어서 다양한 해결방법을 낮은 비용으로 적용할 수 있는 점.
- 공신력있는 기관에서 검증한 견고한 알고리즘
- 유지보수 측면

특히, 대회를 넘어서 인공위성영상 인식 과제는 국가안보에 밀접한 연관이 있을 수 있다고 판단하였다. 또한 객체검출과제에 필요한 개발과정은 상대적으로 복잡한 편이며, 많은 오픈소스들이 다양한 상황에서의 검증은 이루어지지 않았다고 판단했다. 이런 문제의식을 바탕으로 검증된 오픈소스를 활용하고자 하였다.

detectron2는 rotated bbox에 대한 공식적인 지원은 하지 않고 있습니다. 따라서 rotated bbox를 처리하기 위한 파이프라인을 구성해야 합니다.

#### 데이터 전처리

horizontal bbox와 다르게 rotated bbox는 transforms.apply_rotated_box를 적용해야한다. 이는 두 박스간의 연산이 근본적으로 다르기 때문에 detectron2의 내부에서 독립적으로 구현해놓은 상태이다.

```python
def transform_instance_annotations(annotation, transforms, image_size):
    bbox = np.asarray([annotation["bbox"]])
    annotation["bbox"] = transforms.apply_rotated_box(bbox)[0]
    annotation["bbox_mode"] = BoxMode.XYWHA_ABS
    return annotation
```


#### 데이터 어그멘테이션

[imgaug](https://imgaug.readthedocs.io/en/latest/index.html)를 활용하여 적용하였다.. 중요한 점은 rotated bbox를 구성하는 4개의 점을 key point라고 해석하여, augmentation을 image와 rbox에 모두 적용하였다.



**1. bbox2keypoint**

아래는 annotation(bbox)를 keypoint로 변환하는 과정 중 일부이다. 

```python
    def _get_keypoints(self, annos, shape):
        """
        Args:
        annos (dict)
        shape (np.ndarray)

        Returns:
        keypoints (imgaug.augmentables.KeypointsOnImage)
        """
        kps, points = [], []
        for anno in annos:
            bbox = self._bbox_cvt1(*anno["bbox"])  
            horizon_bbox_points = self.rb_cvt.bbox_to_points(
            np.array(bbox[:4]))
            rotated_bbox_points = self.rb_cvt.rotate_horizon_bbox_with_theta(
            horizon_bbox_points, bbox[-1]
            )  # radian
            p1 = tuple(rotated_bbox_points[0][:-1])
            p2 = tuple(rotated_bbox_points[1][:-1])
            p3 = tuple(rotated_bbox_points[2][:-1])
            p4 = tuple(rotated_bbox_points[3][:-1])
            points += [p1, p2, p3, p4]
            kps = KeypointsOnImage([Keypoint(x=p[0], y=p[1])
                                    for p in points], shape=shape)
            assert len(kps) % 4 == 0, "Wrong keypoints"
        return kps
```

**2. augmentation**

위의 과정에서 얻은 keypoint를 augmentation 함수에 넣어준다.

```python
image, kps = self.augmentation(image=image, keypoints=kps)
```

**3. keypoint2annotation**

아래의 과정은 keypoint를 다시 annotation 형태로 바꿔주는 함수이다. 주의 할 점은 p1, p2, p3, p4의 순서가 유지되어야 한다는 점이다.

```python
	def _get_rbox(self, kps):
        """
        Args:
            kps (imgaug.augmentables.KeypointsOnImage)

        Returns
            rboxes (List of [center_x, center_y, bbox_width, bbox_height, theta(degree)])
        """
        stack, rboxes = [], []
        for i in range(len(kps)):
            stack.append([kps[i].x, kps[i].y, 1])
            if len(stack) == 4:
                p1, p2, p3, p4 = stack
                while p1[0] != np.min([p1[0], p2[0], p3[0], p4[0]]):
                    p1, p2, p3, p4 = p2, p3, p4, p1
                [xmin, ymin, xmax, ymax], theta = self.rb_cvt.get_rotated_bbox(
                    np.array([p1, p2, p3, p4])
                )
                rbox = self._bbox_cvt2(xmin, ymin, xmax, ymax, theta)
                rboxes.append(rbox)
                stack = []
        assert not stack, "stack {}".format(stack)
        return rboxes
```







### 모델

- ??

### 학습

- ??

### 실험

- Martin 작성

## 결론

- ??

## Ablation Study

- ??

## Reference

- ??