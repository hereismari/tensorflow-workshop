       гK"	  @{R╓Abrain.Event:2жk'~     ╘=d	ўy}{R╓A"А№

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@global_step*
dtype0	
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
┤
enqueue_input/fifo_queueFIFOQueueV2*
	container *
shared_name *
_output_shapes
: *
component_types
2	*"
shapes
: ::
*
capacityш
^
enqueue_input/PlaceholderPlaceholder*
shape:*
dtype0	*
_output_shapes
:
`
enqueue_input/Placeholder_1Placeholder*
shape:*
dtype0*
_output_shapes
:
`
enqueue_input/Placeholder_2Placeholder*
shape:*
dtype0*
_output_shapes
:
▄
$enqueue_input/fifo_queue_EnqueueManyQueueEnqueueManyV2enqueue_input/fifo_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2*

timeout_ms         *
Tcomponents
2	
g
enqueue_input/fifo_queue_CloseQueueCloseV2enqueue_input/fifo_queue*
cancel_pending_enqueues( 
i
 enqueue_input/fifo_queue_Close_1QueueCloseV2enqueue_input/fifo_queue*
cancel_pending_enqueues(
^
enqueue_input/fifo_queue_SizeQueueSizeV2enqueue_input/fifo_queue*
_output_shapes
: 
U
enqueue_input/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 
m
enqueue_input/subSubenqueue_input/fifo_queue_Sizeenqueue_input/sub/y*
_output_shapes
: *
T0
Y
enqueue_input/Maximum/xConst*
value	B : *
dtype0*
_output_shapes
: 
m
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub*
_output_shapes
: *
T0
a
enqueue_input/CastCastenqueue_input/Maximum*

SrcT0*

DstT0*
_output_shapes
: 
X
enqueue_input/mul/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
b
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y*
_output_shapes
: *
T0
╓
Menqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsConst*Y
valuePBN BHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full*
dtype0*
_output_shapes
: 
▄
Henqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullScalarSummaryMenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsenqueue_input/mul*
_output_shapes
: *
T0
[
fifo_queue_DequeueUpTo/nConst*
value
B :А*
dtype0*
_output_shapes
: 
ъ
fifo_queue_DequeueUpToQueueDequeueUpToV2enqueue_input/fifo_queuefifo_queue_DequeueUpTo/n*

timeout_ms         *Q
_output_shapes?
=:         :         :         
*
component_types
2	
й
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
У
,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *nзо╜* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
У
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *nзо=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ё
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
╥
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
▐
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
│
conv2d/kernel
VariableV2*
	container *
shape: *
shared_name *&
_output_shapes
: * 
_class
loc:@conv2d/kernel*
dtype0
╙
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(
А
conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
К
conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ч
conv2d/bias
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@conv2d/bias*
dtype0
╢
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
n
conv2d/bias/readIdentityconv2d/bias*
_class
loc:@conv2d/bias*
_output_shapes
: *
T0
q
conv2d/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
q
 conv2d/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
▄
conv2d/convolutionConv2Dfifo_queue_DequeueUpTo:1conv2d/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*/
_output_shapes
:          *
T0
Р
conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*/
_output_shapes
:          *
data_formatNHWC*
T0
]
conv2d/ReluReluconv2d/BiasAdd*/
_output_shapes
:          *
T0
║
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*/
_output_shapes
:          *
data_formatNHWC*
paddingVALID*
strides
*
T0
н
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
Ч
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *═╠L╜*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ч
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *═╠L=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ў
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
┌
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
Ї
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
╖
conv2d_1/kernel
VariableV2*
	container *
shape: @*
shared_name *&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel*
dtype0
█
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
О
conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
Ы
conv2d_1/bias
VariableV2*
	container *
shape:@*
shared_name *
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
dtype0
╛
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
s
conv2d_2/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
▌
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*/
_output_shapes
:         @*
T0
Ц
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*/
_output_shapes
:         @*
data_formatNHWC*
T0
a
conv2d_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:         @*
T0
╛
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*/
_output_shapes
:         @*
data_formatNHWC*
paddingVALID*
strides
*
T0
d
flatten/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
_output_shapes
:*
T0
e
flatten/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
е
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
:*
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
W
flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
x
flatten/ProdProdflatten/strided_sliceflatten/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Z
flatten/stack/0Const*
valueB :
         *
dtype0*
_output_shapes
: 
n
flatten/stackPackflatten/stack/0flatten/Prod*
N*
_output_shapes
:*

axis *
T0
Г
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0*(
_output_shapes
:         А*
T0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *╫│]╜*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *╫│]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ч
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
АА*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
т
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
╘
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
е
dense/kernel
VariableV2*
	container *
shape:
АА*
shared_name * 
_output_shapes
:
АА*
_class
loc:@dense/kernel*
dtype0
╔
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
К
dense/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:А
Ч

dense/bias
VariableV2*
	container *
shape:А*
shared_name *
_output_shapes	
:А*
_class
loc:@dense/bias*
dtype0
│
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:А*
T0
У
dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_a( *(
_output_shapes
:         А*
transpose_b( *
T0
Б
dense/BiasAddBiasAdddense/MatMuldense/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:         А*
T0
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *шЬ╜*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *шЬ=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ь
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
щ
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
█
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
з
dense_1/kernel
VariableV2*
	container *
shape:	А
*
shared_name *
_output_shapes
:	А
*!
_class
loc:@dense_1/kernel*
dtype0
╨
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
М
dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

Щ
dense_1/bias
VariableV2*
	container *
shape:
*
shared_name *
_output_shapes
:
*
_class
loc:@dense_1/bias*
dtype0
║
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:
*
use_locking(*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:
*
T0
С
dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_a( *'
_output_shapes
:         
*
transpose_b( *
T0
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*'
_output_shapes
:         
*
data_formatNHWC*
T0
]
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*'
_output_shapes
:         
*
T0
В
softmax_cross_entropy_loss/CastCastfifo_queue_DequeueUpTo:2*

SrcT0*

DstT0*'
_output_shapes
:         

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
o
 softmax_cross_entropy_loss/ShapeShapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
q
"softmax_cross_entropy_loss/Shape_1Shapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Л
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0
И
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
N*
_output_shapes
:*

axis *
T0
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╬
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▌
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
к
"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*0
_output_shapes
:                  *
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Б
"softmax_cross_entropy_loss/Shape_2Shapesoftmax_cross_entropy_loss/Cast*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
П
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0
М
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*
_output_shapes
:*

axis *
T0
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╘
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
х
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
╛
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*0
_output_shapes
:                  *
T0
╪
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:         :                  *
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Н
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Л
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
_output_shapes
:*

axis *
T0
█
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:         *
T0
┤
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*#
_output_shapes
:         *
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
а
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
╣
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:         *
T0
╕
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
е
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
┴
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
о
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
─
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
╔
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ы
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ь
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ъ
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
М
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
щ
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
┐
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
ц
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
╟
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0
р
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
─
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
╙
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
│
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
й
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
╖
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
╡
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
╗
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
╜
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
е
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
╝
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
П
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
╕
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
а
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
╫#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
н
OptimizeLoss/learning_rate
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0
ю
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*-
_class#
!loc:@OptimizeLoss/learning_rate*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
Ч
OptimizeLoss/learning_rate/readIdentityOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
_output_shapes
: *
T0
_
OptimizeLoss/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
OptimizeLoss/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0
М
GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
·
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
у
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ї
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0
Г
@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Е
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ш
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ы
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
■
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0
И
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
╙
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
┘
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Г
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
Ы
@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Д
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
▀
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
щ
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
Н
JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Я
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
О
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
П
AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0
Н
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▓
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualUOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1HOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
┤
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/EqualHOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeUOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
ц
NOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectG^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1
є
VOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectO^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
∙
XOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1IdentityFOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1O^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*Y
_classO
MKloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0
Т
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Н
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
Ю
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
out_type0*
_output_shapes
:*
T0
Н
?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Shape*

Tmultiples0*#
_output_shapes
:         *
T0
д
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
Е
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▄
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:         *
T0
Ч
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Л
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:         *
T0
▄
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*#
_output_shapes
:         *
T0
Э
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Д
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
▀
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
Ў
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:         *
T0
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
Ъ
POptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
┤
JOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeXOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1POptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
└
HOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
е
GOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/TileTileJOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeHOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*#
_output_shapes
:         *
T0
Э
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
▐
\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
·
jOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ъ
XOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMulGOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
х
XOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumXOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/muljOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
╠
\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeXOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
З
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/SelectGOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:         *
T0
ы
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1lOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
▀
^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:         *
T0
н
eOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOp]^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
╤
mOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentity\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshapef^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*o
_classe
caloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
ф
oOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Identity^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1f^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*q
_classg
ecloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:         *
T0
о
dOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
■
bOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumoOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1dOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
й
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0
м
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:         *
T0
Р
!OptimizeLoss/gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:                  *
T0
Щ
NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
░
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
°
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:                  *
T0
У
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
Ь
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:         
*
T0
┴
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*'
_output_shapes
:         
*
T0
Л
AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ї
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*#
_output_shapes
:         *
T0
К
9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ъ
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*'
_output_shapes
:         *
T0
х
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*'
_output_shapes
:         
*
T0
м
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*'
_output_shapes
:         
*
T0
╡
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:
*
data_formatNHWC*
T0
▓
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
║
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*'
_output_shapes
:         
*
T0
╗
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
я
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_a( *(
_output_shapes
:         А*
transpose_b(*
T0
▀
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	А
*
transpose_b( *
T0
н
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
╣
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
╢
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	А
*
T0
┐
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu*(
_output_shapes
:         А*
T0
▓
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
_output_shapes	
:А*
data_formatNHWC*
T0
м
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
│
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
┤
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
щ
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_a( *(
_output_shapes
:         А*
transpose_b(*
T0
с
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
АА*
transpose_b( *
T0
з
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
▒
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
п
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
АА*
T0
И
1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
_output_shapes
:*
T0
№
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:         @*
T0
╕
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:         @*
data_formatNHWC*
paddingVALID*
strides
*
T0
╚
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*/
_output_shapes
:         @*
T0
╖
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
╡
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
╞
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*/
_output_shapes
:         @*
T0
┐
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
Л
6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeShapemax_pooling2d/MaxPool*
out_type0*
_output_shapes
:*
T0
Э
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*J
_output_shapes8
6:4                                    *
T0
С
8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
■
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: @*
T0
╪
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
Є
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:          *
T0
э
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
╚
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*/
_output_shapes
:          *
data_formatNHWC*
paddingVALID*
strides
*
T0
┬
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*/
_output_shapes
:          *
T0
│
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
п
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
╛
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:          *
T0
╖
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
М
4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeShapefifo_queue_DequeueUpTo:1*
out_type0*
_output_shapes
:*
T0
Х
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*J
_output_shapes8
6:4                                    *
T0
П
6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:
√
COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterfifo_queue_DequeueUpTo:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: *
T0
╥
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
ъ
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
х
IOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1IdentityCOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
h
OptimizeLoss/loss/tagsConst*"
valueB BOptimizeLoss/loss*
dtype0*
_output_shapes
: 
}
OptimizeLoss/lossScalarSummaryOptimizeLoss/loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
╡
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
use_locking( *
T0
Я
:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
_output_shapes
: *
use_locking( *
T0
╜
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
use_locking( *
T0
з
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
use_locking( *
T0
ж
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
use_locking( *
T0
Ь
9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
_output_shapes	
:А*
use_locking( *
T0
н
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
use_locking( *
T0
г
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0
Х
OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent
Ц
OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ъ
OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
_output_shapes
: *
use_locking( *
T0	
╕
OptimizeLoss/control_dependencyIdentity softmax_cross_entropy_loss/value^OptimizeLoss/train*3
_class)
'%loc:@softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
m
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*#
_output_shapes
:         *
T0
U
SoftmaxSoftmaxdense_2/Softmax*'
_output_shapes
:         
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
q
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*#
_output_shapes
:         *
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
z
ArgMax_2ArgMaxfifo_queue_DequeueUpTo:2ArgMax_2/dimension*

Tidx0*#
_output_shapes
:         *
T0
P
EqualEqualArgMax_2ArgMax_1*#
_output_shapes
:         *
T0	
S
ToFloatCastEqual*

SrcT0
*

DstT0*#
_output_shapes
:         
S
accuracy/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
accuracy/total
VariableV2*
	container *
dtype0*
shape: *
shared_name *
_output_shapes
: 
м
accuracy/total/AssignAssignaccuracy/totalaccuracy/zeros*!
_class
loc:@accuracy/total*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
s
accuracy/total/readIdentityaccuracy/total*!
_class
loc:@accuracy/total*
_output_shapes
: *
T0
U
accuracy/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
r
accuracy/count
VariableV2*
	container *
dtype0*
shape: *
shared_name *
_output_shapes
: 
о
accuracy/count/AssignAssignaccuracy/countaccuracy/zeros_1*!
_class
loc:@accuracy/count*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
s
accuracy/count/readIdentityaccuracy/count*!
_class
loc:@accuracy/count*
_output_shapes
: *
T0
O
accuracy/SizeSizeToFloat*
out_type0*
_output_shapes
: *
T0
Y
accuracy/ToFloat_1Castaccuracy/Size*

SrcT0*

DstT0*
_output_shapes
: 
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
j
accuracy/SumSumToFloataccuracy/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Ф
accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
_output_shapes
: *
use_locking( *
T0
ж
accuracy/AssignAdd_1	AssignAddaccuracy/countaccuracy/ToFloat_1^ToFloat*!
_class
loc:@accuracy/count*
_output_shapes
: *
use_locking( *
T0
W
accuracy/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
accuracy/GreaterGreateraccuracy/count/readaccuracy/Greater/y*
_output_shapes
: *
T0
f
accuracy/truedivRealDivaccuracy/total/readaccuracy/count/read*
_output_shapes
: *
T0
U
accuracy/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
accuracy/valueSelectaccuracy/Greateraccuracy/truedivaccuracy/value/e*
_output_shapes
: *
T0
Y
accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
accuracy/Greater_1Greateraccuracy/AssignAdd_1accuracy/Greater_1/y*
_output_shapes
: *
T0
h
accuracy/truediv_1RealDivaccuracy/AssignAddaccuracy/AssignAdd_1*
_output_shapes
: *
T0
Y
accuracy/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
accuracy/update_opSelectaccuracy/Greater_1accuracy/truediv_1accuracy/update_op/e*
_output_shapes
: *
T0
O

mean/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

mean/total
VariableV2*
	container *
dtype0*
shape: *
shared_name *
_output_shapes
: 
Ь
mean/total/AssignAssign
mean/total
mean/zeros*
_class
loc:@mean/total*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
g
mean/total/readIdentity
mean/total*
_class
loc:@mean/total*
_output_shapes
: *
T0
Q
mean/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
n

mean/count
VariableV2*
	container *
dtype0*
shape: *
shared_name *
_output_shapes
: 
Ю
mean/count/AssignAssign
mean/countmean/zeros_1*
_class
loc:@mean/count*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
g
mean/count/readIdentity
mean/count*
_class
loc:@mean/count*
_output_shapes
: *
T0
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*

DstT0*
_output_shapes
: 
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
{
mean/SumSum softmax_cross_entropy_loss/value
mean/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Д
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_class
loc:@mean/total*
_output_shapes
: *
use_locking( *
T0
п
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1!^softmax_cross_entropy_loss/value*
_class
loc:@mean/count*
_output_shapes
: *
use_locking( *
T0
S
mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
mean/GreaterGreatermean/count/readmean/Greater/y*
_output_shapes
: *
T0
Z
mean/truedivRealDivmean/total/readmean/count/read*
_output_shapes
: *
T0
Q
mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_

mean/valueSelectmean/Greatermean/truedivmean/value/e*
_output_shapes
: *
T0
U
mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
mean/Greater_1Greatermean/AssignAdd_1mean/Greater_1/y*
_output_shapes
: *
T0
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
_output_shapes
: *
T0
U
mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
mean/update_opSelectmean/Greater_1mean/truediv_1mean/update_op/e*
_output_shapes
: *
T0
8

group_depsNoOp^accuracy/update_op^mean/update_op
{
eval_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Л
	eval_step
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@eval_step*
dtype0	
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
_class
loc:@eval_step*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
T0	
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
_output_shapes
: *
use_locking( *
T0	
∙
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
е
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
б
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
й
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
е
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
г
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Я
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
з
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
г
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
┐
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
▒
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_14*
N*
_output_shapes
:*

axis *
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
╬
$report_uninitialized_variables/ConstConst*ї
valueыBшBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask*
_output_shapes
:*
end_mask *
shrink_axis_mask *
new_axis_mask *
T0
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
: *
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
_output_shapes
:*

axis *
T0
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0

Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:         *
T0	
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
з
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
г
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
л
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
з
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
е
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
б
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
й
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
е
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
┴
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
к
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_9*
N
*
_output_shapes
:
*

axis *
T0

}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:

Н
&report_uninitialized_variables_1/ConstConst*▓
valueиBе
Bglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rate*
dtype0*
_output_shapes
:

}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask*
_output_shapes
:*
end_mask *
shrink_axis_mask *
new_axis_mask *
T0
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
: *
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
_output_shapes
:*

axis *
T0
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0

Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:         *
T0	
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:         
y
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
а
Merge/MergeSummaryMergeSummaryHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_796ba6372d62457891ce9e323aab0162/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
 
save/SaveV2/tensor_namesConst*▓
valueиBе
BOptimizeLoss/learning_rateBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:

w
save/SaveV2/shape_and_slicesConst*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

Ъ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*

axis *
T0
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst*/
value&B$BOptimizeLoss/learning_rate*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/AssignAssignOptimizeLoss/learning_ratesave/RestoreV2*-
_class#
!loc:@OptimizeLoss/learning_rate*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
q
save/RestoreV2_1/tensor_namesConst* 
valueBBconv2d/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
д
save/Assign_1Assignconv2d/biassave/RestoreV2_1*
_class
loc:@conv2d/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
s
save/RestoreV2_2/tensor_namesConst*"
valueBBconv2d/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
┤
save/Assign_2Assignconv2d/kernelsave/RestoreV2_2* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(
s
save/RestoreV2_3/tensor_namesConst*"
valueBBconv2d_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
и
save/Assign_3Assignconv2d_1/biassave/RestoreV2_3* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
u
save/RestoreV2_4/tensor_namesConst*$
valueBBconv2d_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_4Assignconv2d_1/kernelsave/RestoreV2_4*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(
p
save/RestoreV2_5/tensor_namesConst*
valueBB
dense/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
г
save/Assign_5Assign
dense/biassave/RestoreV2_5*
_class
loc:@dense/bias*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(
r
save/RestoreV2_6/tensor_namesConst*!
valueBBdense/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
м
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(
r
save/RestoreV2_7/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ж
save/Assign_7Assigndense_1/biassave/RestoreV2_7*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:
*
use_locking(*
validate_shape(
t
save/RestoreV2_8/tensor_namesConst*#
valueBBdense_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
п
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(
q
save/RestoreV2_9/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
а
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
_class
loc:@global_step*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
╕
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"/Ц▄╙ОЮ     K╘ я	╛oК{R╓AJБ╜
Я1∙0
9
Add
x"T
y"T
z"T"
Ttype:
2	
А
ApplyGradientDescent
var"TА

alpha"T

delta"T
out"TА"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignAdd
ref"TА

value"T

output_ref"TА"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
╚
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ю
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
э
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
о
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint         "
	containerstring "
shared_namestring И
4
Fill
dims

value"T
output"T"	
Ttype
М
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
┼
MaxPool

input"T
output"T"
Ttype0:
2		"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ы
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
8
MergeSummary
inputs*N
summary"
Nint(0
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
B
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( И
М
QueueDequeueUpToV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint         И
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint         И
&
QueueSizeV2

handle
sizeИ
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*	1.2.0-rc12v1.2.0-rc0-24-g94484aaА№

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@global_step*
dtype0	
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_class
loc:@global_step*
use_locking(*
_output_shapes
: *
T0	
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
┤
enqueue_input/fifo_queueFIFOQueueV2*
	container *
shared_name *
_output_shapes
: *
component_types
2	*"
shapes
: ::
*
capacityш
^
enqueue_input/PlaceholderPlaceholder*
shape:*
dtype0	*
_output_shapes
:
`
enqueue_input/Placeholder_1Placeholder*
shape:*
dtype0*
_output_shapes
:
`
enqueue_input/Placeholder_2Placeholder*
shape:*
dtype0*
_output_shapes
:
▄
$enqueue_input/fifo_queue_EnqueueManyQueueEnqueueManyV2enqueue_input/fifo_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2*

timeout_ms         *
Tcomponents
2	
g
enqueue_input/fifo_queue_CloseQueueCloseV2enqueue_input/fifo_queue*
cancel_pending_enqueues( 
i
 enqueue_input/fifo_queue_Close_1QueueCloseV2enqueue_input/fifo_queue*
cancel_pending_enqueues(
^
enqueue_input/fifo_queue_SizeQueueSizeV2enqueue_input/fifo_queue*
_output_shapes
: 
U
enqueue_input/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 
m
enqueue_input/subSubenqueue_input/fifo_queue_Sizeenqueue_input/sub/y*
_output_shapes
: *
T0
Y
enqueue_input/Maximum/xConst*
value	B : *
dtype0*
_output_shapes
: 
m
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub*
_output_shapes
: *
T0
a
enqueue_input/CastCastenqueue_input/Maximum*

SrcT0*

DstT0*
_output_shapes
: 
X
enqueue_input/mul/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
b
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y*
_output_shapes
: *
T0
╓
Menqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsConst*Y
valuePBN BHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full*
dtype0*
_output_shapes
: 
▄
Henqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullScalarSummaryMenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsenqueue_input/mul*
_output_shapes
: *
T0
[
fifo_queue_DequeueUpTo/nConst*
value
B :А*
dtype0*
_output_shapes
: 
ъ
fifo_queue_DequeueUpToQueueDequeueUpToV2enqueue_input/fifo_queuefifo_queue_DequeueUpTo/n*

timeout_ms         *
component_types
2	*Q
_output_shapes?
=:         :         :         

й
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
У
,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *nзо╜* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
У
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *nзо=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ё
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
╥
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
▐
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
│
conv2d/kernel
VariableV2*
	container *
shape: *
shared_name *&
_output_shapes
: * 
_class
loc:@conv2d/kernel*
dtype0
╙
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0
А
conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
К
conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ч
conv2d/bias
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@conv2d/bias*
dtype0
╢
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
_output_shapes
: *
T0
n
conv2d/bias/readIdentityconv2d/bias*
_class
loc:@conv2d/bias*
_output_shapes
: *
T0
q
conv2d/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
q
 conv2d/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
▄
conv2d/convolutionConv2Dfifo_queue_DequeueUpTo:1conv2d/kernel/read*
strides
*/
_output_shapes
:          *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Р
conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
data_formatNHWC*/
_output_shapes
:          *
T0
]
conv2d/ReluReluconv2d/BiasAdd*/
_output_shapes
:          *
T0
║
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*/
_output_shapes
:          *
data_formatNHWC*
paddingVALID*
strides
*
T0
н
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
Ч
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *═╠L╜*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ч
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *═╠L=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ў
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
┌
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
Ї
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
╖
conv2d_1/kernel
VariableV2*
	container *
shape: @*
shared_name *&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel*
dtype0
█
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*&
_output_shapes
: @*
T0
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
О
conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
Ы
conv2d_1/bias
VariableV2*
	container *
shape:@*
shared_name *
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
dtype0
╛
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
_output_shapes
:@*
T0
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
s
conv2d_2/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
▌
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*/
_output_shapes
:         @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ц
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
data_formatNHWC*/
_output_shapes
:         @*
T0
a
conv2d_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:         @*
T0
╛
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*/
_output_shapes
:         @*
data_formatNHWC*
paddingVALID*
strides
*
T0
d
flatten/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
_output_shapes
:*
T0
e
flatten/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
е
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
:*
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
W
flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
x
flatten/ProdProdflatten/strided_sliceflatten/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Z
flatten/stack/0Const*
valueB :
         *
dtype0*
_output_shapes
: 
n
flatten/stackPackflatten/stack/0flatten/Prod*
N*

axis *
_output_shapes
:*
T0
Г
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0*(
_output_shapes
:         А*
T0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *╫│]╜*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *╫│]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ч
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
АА*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
т
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
╘
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
е
dense/kernel
VariableV2*
	container *
shape:
АА*
shared_name * 
_output_shapes
:
АА*
_class
loc:@dense/kernel*
dtype0
╔
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
АА*
T0
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
T0
К
dense/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:А
Ч

dense/bias
VariableV2*
	container *
shape:А*
shared_name *
_output_shapes	
:А*
_class
loc:@dense/bias*
dtype0
│
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:А*
T0
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:А*
T0
У
dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:         А*
T0
Б
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:         А*
T0
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *шЬ╜*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *шЬ=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ь
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
щ
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
█
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
з
dense_1/kernel
VariableV2*
	container *
shape:	А
*
shared_name *
_output_shapes
:	А
*!
_class
loc:@dense_1/kernel*
dtype0
╨
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	А
*
T0
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А
*
T0
М
dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

Щ
dense_1/bias
VariableV2*
	container *
shape:
*
shared_name *
_output_shapes
:
*
_class
loc:@dense_1/bias*
dtype0
║
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
_output_shapes
:
*
T0
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:
*
T0
С
dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *'
_output_shapes
:         
*
T0
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:         
*
T0
]
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*'
_output_shapes
:         
*
T0
В
softmax_cross_entropy_loss/CastCastfifo_queue_DequeueUpTo:2*

SrcT0*

DstT0*'
_output_shapes
:         

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
o
 softmax_cross_entropy_loss/ShapeShapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
q
"softmax_cross_entropy_loss/Shape_1Shapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Л
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0
И
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
N*

axis *
_output_shapes
:*
T0
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╬
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▌
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
к
"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*0
_output_shapes
:                  *
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Б
"softmax_cross_entropy_loss/Shape_2Shapesoftmax_cross_entropy_loss/Cast*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
П
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0
М
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*

axis *
_output_shapes
:*
T0
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╘
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
х
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
╛
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*0
_output_shapes
:                  *
T0
╪
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:         :                  *
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Н
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Л
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*

axis *
_output_shapes
:*
T0
█
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:         *
T0
┤
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*#
_output_shapes
:         *
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
а
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
╣
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:         *
T0
╕
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
е
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
┴
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
о
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
─
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
╔
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ы
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ь
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ъ
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
М
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
щ
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
┐
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
ц
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
╟
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0
р
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
─
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
╙
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
│
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
й
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
╖
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
╡
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
╗
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
╜
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
е
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
╝
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
П
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
╕
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
а
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
╫#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
н
OptimizeLoss/learning_rate
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0
ю
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*
validate_shape(*-
_class#
!loc:@OptimizeLoss/learning_rate*
use_locking(*
_output_shapes
: *
T0
Ч
OptimizeLoss/learning_rate/readIdentityOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
_output_shapes
: *
T0
_
OptimizeLoss/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
OptimizeLoss/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0
М
GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
·
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
у
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ї
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0
Г
@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Е
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ш
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ы
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
■
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0
И
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
╙
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
┘
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Г
>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
Ы
@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Д
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
▀
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
щ
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
Н
JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Я
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
О
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
П
AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0
Н
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▓
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualUOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1HOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
┤
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/EqualHOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeUOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
ц
NOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectG^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1
є
VOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/SelectO^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
∙
XOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1IdentityFOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1O^OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*Y
_classO
MKloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0
Т
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Н
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
Ю
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
out_type0*
_output_shapes
:*
T0
Н
?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Shape*

Tmultiples0*#
_output_shapes
:         *
T0
д
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
Е
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▄
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:         *
T0
Ч
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Л
BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:         *
T0
▄
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*#
_output_shapes
:         *
T0
Э
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Д
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
▀
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
Ў
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:         *
T0
я
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
Ъ
POptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
┤
JOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeXOptimizeLoss/gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1POptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
└
HOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
е
GOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/TileTileJOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeHOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*#
_output_shapes
:         *
T0
Э
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
▐
\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
·
jOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ъ
XOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMulGOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
х
XOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumXOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/muljOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
╠
\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeXOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
З
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/SelectGOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:         *
T0
ы
ZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1lOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
▀
^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeZOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:         *
T0
н
eOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOp]^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
╤
mOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentity\OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshapef^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*o
_classe
caloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
ф
oOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Identity^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1f^OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*q
_classg
ecloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:         *
T0
о
dOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
■
bOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumoOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1dOptimizeLoss/gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
й
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0
м
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:         *
T0
Р
!OptimizeLoss/gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:                  *
T0
Щ
NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
░
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:         *
T0
°
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:                  *
T0
У
DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapedense_2/Softmax*
out_type0*
_output_shapes
:*
T0
Ь
FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:         
*
T0
┴
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*'
_output_shapes
:         
*
T0
Л
AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ї
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*#
_output_shapes
:         *
T0
К
9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ъ
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*'
_output_shapes
:         *
T0
х
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*'
_output_shapes
:         
*
T0
м
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*'
_output_shapes
:         
*
T0
╡
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:
*
T0
▓
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
║
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*'
_output_shapes
:         
*
T0
╗
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
я
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
transpose_a( *(
_output_shapes
:         А*
T0
▀
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	А
*
T0
н
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
╣
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
╢
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	А
*
T0
┐
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu*(
_output_shapes
:         А*
T0
▓
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
м
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
│
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
┤
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
щ
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
transpose_a( *(
_output_shapes
:         А*
T0
с
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(* 
_output_shapes
:
АА*
T0
з
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
▒
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
п
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
АА*
T0
И
1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
_output_shapes
:*
T0
№
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:         @*
T0
╕
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:         @*
data_formatNHWC*
paddingVALID*
strides
*
T0
╚
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*/
_output_shapes
:         @*
T0
╖
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
╡
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
╞
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*/
_output_shapes
:         @*
T0
┐
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
Л
6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeShapemax_pooling2d/MaxPool*
out_type0*
_output_shapes
:*
T0
Э
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
С
8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
■
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
╪
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
Є
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:          *
T0
э
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
╚
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*/
_output_shapes
:          *
data_formatNHWC*
paddingVALID*
strides
*
T0
┬
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*/
_output_shapes
:          *
T0
│
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
п
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
╛
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:          *
T0
╖
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
М
4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeShapefifo_queue_DequeueUpTo:1*
out_type0*
_output_shapes
:*
T0
Х
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
П
6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:
√
COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterfifo_queue_DequeueUpTo:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
╥
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
ъ
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
х
IOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1IdentityCOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
h
OptimizeLoss/loss/tagsConst*"
valueB BOptimizeLoss/loss*
dtype0*
_output_shapes
: 
}
OptimizeLoss/lossScalarSummaryOptimizeLoss/loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
╡
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*
use_locking( *&
_output_shapes
: *
T0
Я
:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
use_locking( *
_output_shapes
: *
T0
╜
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*
use_locking( *&
_output_shapes
: @*
T0
з
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
use_locking( *
_output_shapes
:@*
T0
ж
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
use_locking( * 
_output_shapes
:
АА*
T0
Ь
9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_locking( *
_output_shapes	
:А*
T0
н
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
use_locking( *
_output_shapes
:	А
*
T0
г
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
use_locking( *
_output_shapes
:
*
T0
Х
OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent
Ц
OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ъ
OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
use_locking( *
_output_shapes
: *
T0	
╕
OptimizeLoss/control_dependencyIdentity softmax_cross_entropy_loss/value^OptimizeLoss/train*3
_class)
'%loc:@softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
m
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*#
_output_shapes
:         *
T0
U
SoftmaxSoftmaxdense_2/Softmax*'
_output_shapes
:         
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
q
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*#
_output_shapes
:         *
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
z
ArgMax_2ArgMaxfifo_queue_DequeueUpTo:2ArgMax_2/dimension*

Tidx0*#
_output_shapes
:         *
T0
P
EqualEqualArgMax_2ArgMax_1*#
_output_shapes
:         *
T0	
S
ToFloatCastEqual*

SrcT0
*

DstT0*#
_output_shapes
:         
S
accuracy/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
accuracy/total
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: 
м
accuracy/total/AssignAssignaccuracy/totalaccuracy/zeros*
validate_shape(*!
_class
loc:@accuracy/total*
use_locking(*
_output_shapes
: *
T0
s
accuracy/total/readIdentityaccuracy/total*!
_class
loc:@accuracy/total*
_output_shapes
: *
T0
U
accuracy/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
r
accuracy/count
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: 
о
accuracy/count/AssignAssignaccuracy/countaccuracy/zeros_1*
validate_shape(*!
_class
loc:@accuracy/count*
use_locking(*
_output_shapes
: *
T0
s
accuracy/count/readIdentityaccuracy/count*!
_class
loc:@accuracy/count*
_output_shapes
: *
T0
O
accuracy/SizeSizeToFloat*
out_type0*
_output_shapes
: *
T0
Y
accuracy/ToFloat_1Castaccuracy/Size*

SrcT0*

DstT0*
_output_shapes
: 
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
j
accuracy/SumSumToFloataccuracy/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Ф
accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
_output_shapes
: *
T0
ж
accuracy/AssignAdd_1	AssignAddaccuracy/countaccuracy/ToFloat_1^ToFloat*!
_class
loc:@accuracy/count*
use_locking( *
_output_shapes
: *
T0
W
accuracy/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
accuracy/GreaterGreateraccuracy/count/readaccuracy/Greater/y*
_output_shapes
: *
T0
f
accuracy/truedivRealDivaccuracy/total/readaccuracy/count/read*
_output_shapes
: *
T0
U
accuracy/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
accuracy/valueSelectaccuracy/Greateraccuracy/truedivaccuracy/value/e*
_output_shapes
: *
T0
Y
accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
accuracy/Greater_1Greateraccuracy/AssignAdd_1accuracy/Greater_1/y*
_output_shapes
: *
T0
h
accuracy/truediv_1RealDivaccuracy/AssignAddaccuracy/AssignAdd_1*
_output_shapes
: *
T0
Y
accuracy/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
accuracy/update_opSelectaccuracy/Greater_1accuracy/truediv_1accuracy/update_op/e*
_output_shapes
: *
T0
O

mean/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

mean/total
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: 
Ь
mean/total/AssignAssign
mean/total
mean/zeros*
validate_shape(*
_class
loc:@mean/total*
use_locking(*
_output_shapes
: *
T0
g
mean/total/readIdentity
mean/total*
_class
loc:@mean/total*
_output_shapes
: *
T0
Q
mean/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
n

mean/count
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: 
Ю
mean/count/AssignAssign
mean/countmean/zeros_1*
validate_shape(*
_class
loc:@mean/count*
use_locking(*
_output_shapes
: *
T0
g
mean/count/readIdentity
mean/count*
_class
loc:@mean/count*
_output_shapes
: *
T0
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*

DstT0*
_output_shapes
: 
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
{
mean/SumSum softmax_cross_entropy_loss/value
mean/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Д
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_class
loc:@mean/total*
use_locking( *
_output_shapes
: *
T0
п
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1!^softmax_cross_entropy_loss/value*
_class
loc:@mean/count*
use_locking( *
_output_shapes
: *
T0
S
mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
mean/GreaterGreatermean/count/readmean/Greater/y*
_output_shapes
: *
T0
Z
mean/truedivRealDivmean/total/readmean/count/read*
_output_shapes
: *
T0
Q
mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_

mean/valueSelectmean/Greatermean/truedivmean/value/e*
_output_shapes
: *
T0
U
mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
mean/Greater_1Greatermean/AssignAdd_1mean/Greater_1/y*
_output_shapes
: *
T0
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
_output_shapes
: *
T0
U
mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
mean/update_opSelectmean/Greater_1mean/truediv_1mean/update_op/e*
_output_shapes
: *
T0
8

group_depsNoOp^accuracy/update_op^mean/update_op
{
eval_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Л
	eval_step
VariableV2*
	container *
shape: *
shared_name *
_output_shapes
: *
_class
loc:@eval_step*
dtype0	
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
_output_shapes
: *
T0	
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
T0	
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
_output_shapes
: *
T0	
∙
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
е
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
б
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
й
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
е
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
г
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Я
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
з
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
г
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
┐
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
▒
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_14*
N*

axis *
_output_shapes
:*
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
╬
$report_uninitialized_variables/ConstConst*ї
valueыBшBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask*
_output_shapes
:*
end_mask *
shrink_axis_mask *
new_axis_mask *
T0
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
: *
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*

axis *
_output_shapes
:*
T0
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0

Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:         *
T0	
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
з
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
г
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
л
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
з
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
е
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
б
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
й
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
е
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
┴
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
к
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_9*
N
*

axis *
_output_shapes
:
*
T0

}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:

Н
&report_uninitialized_variables_1/ConstConst*▓
valueиBе
Bglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rate*
dtype0*
_output_shapes
:

}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
ellipsis_mask *
Index0*

begin_mask*
_output_shapes
:*
end_mask *
shrink_axis_mask *
new_axis_mask *
T0
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *
Index0*

begin_mask *
_output_shapes
: *
end_mask*
shrink_axis_mask *
new_axis_mask *
T0
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*

axis *
_output_shapes
:*
T0
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0

Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:         *
T0	
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:         
y
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
а
Merge/MergeSummaryMergeSummaryHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_796ba6372d62457891ce9e323aab0162/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
 
save/SaveV2/tensor_namesConst*▓
valueиBе
BOptimizeLoss/learning_rateBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:

w
save/SaveV2/shape_and_slicesConst*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

Ъ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*

axis *
_output_shapes
:*
T0
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst*/
value&B$BOptimizeLoss/learning_rate*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/AssignAssignOptimizeLoss/learning_ratesave/RestoreV2*
validate_shape(*-
_class#
!loc:@OptimizeLoss/learning_rate*
use_locking(*
_output_shapes
: *
T0
q
save/RestoreV2_1/tensor_namesConst* 
valueBBconv2d/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
д
save/Assign_1Assignconv2d/biassave/RestoreV2_1*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
_output_shapes
: *
T0
s
save/RestoreV2_2/tensor_namesConst*"
valueBBconv2d/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
┤
save/Assign_2Assignconv2d/kernelsave/RestoreV2_2*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0
s
save/RestoreV2_3/tensor_namesConst*"
valueBBconv2d_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
и
save/Assign_3Assignconv2d_1/biassave/RestoreV2_3*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
_output_shapes
:@*
T0
u
save/RestoreV2_4/tensor_namesConst*$
valueBBconv2d_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_4Assignconv2d_1/kernelsave/RestoreV2_4*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*&
_output_shapes
: @*
T0
p
save/RestoreV2_5/tensor_namesConst*
valueBB
dense/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
г
save/Assign_5Assign
dense/biassave/RestoreV2_5*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:А*
T0
r
save/RestoreV2_6/tensor_namesConst*!
valueBBdense/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
м
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
АА*
T0
r
save/RestoreV2_7/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ж
save/Assign_7Assigndense_1/biassave/RestoreV2_7*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
_output_shapes
:
*
T0
t
save/RestoreV2_8/tensor_namesConst*#
valueBBdense_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
п
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	А
*
T0
q
save/RestoreV2_9/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
а
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
validate_shape(*
_class
loc:@global_step*
use_locking(*
_output_shapes
: *
T0	
╕
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard""
ready_op


concat:0" 
global_step

global_step:0"
init_op

group_deps_1"Ю
queue_runnersМЙ
Ж
enqueue_input/fifo_queue$enqueue_input/fifo_queue_EnqueueManyenqueue_input/fifo_queue_Close" enqueue_input/fifo_queue_Close_1*"b
local_variablesO
M
accuracy/total:0
accuracy/count:0
mean/total:0
mean/count:0
eval_step:0"Ь
	variablesОЛ
7
global_step:0global_step/Assignglobal_step/read:0
=
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:0
7
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
d
OptimizeLoss/learning_rate:0!OptimizeLoss/learning_rate/Assign!OptimizeLoss/learning_rate/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"p
	summariesc
a
Jenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full:0
OptimizeLoss/loss:0"
	eval_step

eval_step:0"&

summary_op

Merge/MergeSummary:0"З
trainable_variablesяь
=
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:0
7
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0""
train_op

OptimizeLoss/train"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"!
local_init_op

group_deps_2▌RЛ+       Ж├K	TзК{R╓A*

accuracyд▀>=

loss∙U@╤М{,       Їо╠E	}=ЫКR╓Aш*

accuracy░r?

loss└ц@`╩╚,       Їо╠E	 ЧR╓Aщ*

accuracyЩ*?

loss╬@└a╜к,       Їо╠E	¤1OжR╓A╨*

accuracyZT?

lossш╙?╠ъS