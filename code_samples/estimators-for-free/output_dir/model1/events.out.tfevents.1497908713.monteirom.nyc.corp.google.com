       ŁK"	  @zRÖAbrain.Event:2!ŮaBqý      Âá^ť	uAnzRÖA"äú

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
_output_shapes
: *
capacityč
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*9
_output_shapes'
%:::	
*
component_types
2	
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*'
_output_shapes
: *
data_formatNHWC*
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:@*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*'
_output_shapes
:@*
data_formatNHWC*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b( *
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read* 
_output_shapes
:
*
data_formatNHWC*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	
*
use_locking(*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_a( *
_output_shapes
:	
*
transpose_b( *
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
_output_shapes
:	
*
data_formatNHWC*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
_output_shapes
:*

axis *
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*-
_class#
!loc:@OptimizeLoss/learning_rate*
T0*
_output_shapes
: *
use_locking(*
validate_shape(

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*
_output_shapes
:	*

Tdim0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:
*
data_formatNHWC*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	
*
transpose_b( *
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
transpose_b( *
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: @*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: *
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
use_locking( *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
_output_shapes
: *
use_locking( *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
use_locking( *
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
use_locking( *
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel* 
_output_shapes
:
*
use_locking( *
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
_output_shapes	
:*
use_locking( *
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
use_locking( *
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
_output_shapes
: *
use_locking( *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
_output_shapes
: *
use_locking( *
T0
Ś
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
T0"xă     )WÉi	Xn~zRÖAJö˛
Ł)ý(
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
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
Č
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
î
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
í
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

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
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
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
Ĺ
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
ë
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
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
&
QueueSizeV2

handle
size
ř
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint˙˙˙˙˙˙˙˙˙"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring 
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
ö
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
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *	1.2.0-rc12v1.2.0-rc0-24-g94484aaäú

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
capacityč*
_output_shapes
: 
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2	*9
_output_shapes'
%:::	

Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
data_formatNHWC*'
_output_shapes
: *
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*&
_output_shapes
: @*
T0

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*'
_output_shapes
:@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
data_formatNHWC*'
_output_shapes
:@*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
*
T0
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:*
T0
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_b( *
transpose_a( * 
_output_shapes
:
*
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	
*
T0
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *
_output_shapes
:	
*
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*
_output_shapes
:	
*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*

axis *
_output_shapes
:*
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*
validate_shape(*-
_class#
!loc:@OptimizeLoss/learning_rate*
use_locking(*
_output_shapes
: *
T0

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*
_output_shapes
:	*
T0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:
*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	
*
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(* 
_output_shapes
:
*
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*
use_locking( *&
_output_shapes
: *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
use_locking( *
_output_shapes
: *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*
use_locking( *&
_output_shapes
: @*
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
use_locking( *
_output_shapes
:@*
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
use_locking( * 
_output_shapes
:
*
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_locking( *
_output_shapes	
:*
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
use_locking( *
_output_shapes
:	
*
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
use_locking( *
_output_shapes
:
*
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
use_locking( *
_output_shapes
: *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
_output_shapes
: *
T0
Ś
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
T0""

savers " 
global_step

global_step:0"
trainable_variablesďě
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
OptimizeLoss/train"Ü
queue_runnersĘÇ
Ä
"enqueue_input/random_shuffle_queue.enqueue_input/random_shuffle_queue_EnqueueMany0enqueue_input/random_shuffle_queue_EnqueueMany_10enqueue_input/random_shuffle_queue_EnqueueMany_20enqueue_input/random_shuffle_queue_EnqueueMany_3(enqueue_input/random_shuffle_queue_Close"*enqueue_input/random_shuffle_queue_Close_1*"9
local_variables&
$
accuracy/total:0
accuracy/count:0"
	variables
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
OptimizeLoss/learning_rate:0!OptimizeLoss/learning_rate/Assign!OptimizeLoss/learning_rate/read:0"T
lossesJ
H
"softmax_cross_entropy_loss/value:0
"softmax_cross_entropy_loss/value:0"{
	summariesn
l
Uenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full:0
OptimizeLoss/loss:0B?ĐX     <ČH	Ćś˛zRÖA"°

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
_output_shapes
: *
capacityč
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*9
_output_shapes'
%:::	
*
component_types
2	
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*'
_output_shapes
: *
data_formatNHWC*
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:@*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*'
_output_shapes
:@*
data_formatNHWC*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b( *
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read* 
_output_shapes
:
*
data_formatNHWC*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	
*
use_locking(*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_a( *
_output_shapes
:	
*
transpose_b( *
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
_output_shapes
:	
*
data_formatNHWC*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
_output_shapes
:*

axis *
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*-
_class#
!loc:@OptimizeLoss/learning_rate*
T0*
_output_shapes
: *
use_locking(*
validate_shape(

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*
_output_shapes
:	*

Tdim0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:
*
data_formatNHWC*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	
*
transpose_b( *
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
transpose_b( *
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: @*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: *
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
use_locking( *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
_output_shapes
: *
use_locking( *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
use_locking( *
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
use_locking( *
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel* 
_output_shapes
:
*
use_locking( *
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
_output_shapes	
:*
use_locking( *
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
use_locking( *
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
_output_shapes
: *
use_locking( *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
_output_shapes
: *
use_locking( *
T0
Ś
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
ů
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ą
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
§
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
ż
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11*
N*
_output_shapes
:*

axis *
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ť
$report_uninitialized_variables/ConstConst*Ň
valueČBĹBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/count*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ů
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

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ő
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
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
Ż
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
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0


1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ź
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ą
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ł
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ť
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ą
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Š
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
Á
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
Ş
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


&report_uninitialized_variables_1/ConstConst*˛
value¨BĽ
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

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ă
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

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ű
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

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ë
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
ł
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
ł
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ń
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
á
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0


3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
Ť
Merge/MergeSummaryMergeSummarySenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_fa5c6d76ecb24835901de98868c8b534/part*
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
˙
save/SaveV2/tensor_namesConst*˛
value¨BĽ
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


save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0

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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
´
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_5Assign
dense/biassave/RestoreV2_5*
_class
loc:@dense/bias*
T0*
_output_shapes	
:*
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
 
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
_class
loc:@global_step*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
¸
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"c5$>öx     řŻvĘ	ľIťzRÖAJéń
0ř/
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
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
Č
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
î
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
í
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

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype

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
ref"dtype
is_initialized
"
dtypetype
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
Ĺ
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
ë
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
2	
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
2	
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

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
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
&
QueueSizeV2

handle
size
ř
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint˙˙˙˙˙˙˙˙˙"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring 
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
ö
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 

Where	
input
	
index	*	1.2.0-rc12v1.2.0-rc0-24-g94484aa°

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
capacityč*
_output_shapes
: 
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2	*9
_output_shapes'
%:::	

Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
data_formatNHWC*'
_output_shapes
: *
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*&
_output_shapes
: @*
T0

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*'
_output_shapes
:@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
data_formatNHWC*'
_output_shapes
:@*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
*
T0
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:*
T0
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_b( *
transpose_a( * 
_output_shapes
:
*
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	
*
T0
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *
_output_shapes
:	
*
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*
_output_shapes
:	
*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*

axis *
_output_shapes
:*
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*
validate_shape(*-
_class#
!loc:@OptimizeLoss/learning_rate*
use_locking(*
_output_shapes
: *
T0

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*
_output_shapes
:	*
T0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:
*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	
*
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(* 
_output_shapes
:
*
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*
use_locking( *&
_output_shapes
: *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
use_locking( *
_output_shapes
: *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*
use_locking( *&
_output_shapes
: @*
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
use_locking( *
_output_shapes
:@*
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
use_locking( * 
_output_shapes
:
*
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_locking( *
_output_shapes	
:*
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
use_locking( *
_output_shapes
:	
*
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
use_locking( *
_output_shapes
:
*
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
use_locking( *
_output_shapes
: *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
_output_shapes
: *
T0
Ś
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
ů
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ą
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
§
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
ż
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11*
N*

axis *
_output_shapes
:*
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ť
$report_uninitialized_variables/ConstConst*Ň
valueČBĹBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/count*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ů
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

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ő
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
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
Ż
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
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0


1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ź
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ą
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ł
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ť
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ą
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Š
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
Á
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
Ş
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


&report_uninitialized_variables_1/ConstConst*˛
value¨BĽ
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

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ă
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

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ű
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

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ë
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
ł
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
ł
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ń
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
á
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0


3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
Ť
Merge/MergeSummaryMergeSummarySenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_fa5c6d76ecb24835901de98868c8b534/part*
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
˙
save/SaveV2/tensor_namesConst*˛
value¨BĽ
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


save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0

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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
´
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_5Assign
dense/biassave/RestoreV2_5*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:*
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
*
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
 
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
validate_shape(*
_class
loc:@global_step*
use_locking(*
_output_shapes
: *
T0	
¸
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
ready_op


concat:0" 
global_step

global_step:0"
init_op


group_deps"Ü
queue_runnersĘÇ
Ä
"enqueue_input/random_shuffle_queue.enqueue_input/random_shuffle_queue_EnqueueMany0enqueue_input/random_shuffle_queue_EnqueueMany_10enqueue_input/random_shuffle_queue_EnqueueMany_20enqueue_input/random_shuffle_queue_EnqueueMany_3(enqueue_input/random_shuffle_queue_Close"*enqueue_input/random_shuffle_queue_Close_1*"9
local_variables&
$
accuracy/total:0
accuracy/count:0"
	variables
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
OptimizeLoss/learning_rate:0!OptimizeLoss/learning_rate/Assign!OptimizeLoss/learning_rate/read:0"T
lossesJ
H
"softmax_cross_entropy_loss/value:0
"softmax_cross_entropy_loss/value:0"&

summary_op

Merge/MergeSummary:0"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
trainable_variablesďě
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
save/Const:0save/Identity:0save/restore_all (5 @F8"{
	summariesn
l
Uenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full:0
OptimizeLoss/loss:0"!
local_init_op

group_deps_1KĹýÉ-       <Aű	{ÄÖzRÖA: output_dir/model1/model.ckpt,á7Ü       mS+		ĆÖzRÖA:ŤĘ=       łąé	ňÇÖzRÖA*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossY@ęá%       ęź6ó	čü|RÖAe*

global_step/secgC:ABÝ`!       łąé	Dü|RÖAe*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossÖ5@QÎać&       sOă 	ti~RÖAÉ*

global_step/secOxAfWáŤ       Aď	Ăi~RÖAÉ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossh@!Ű&       sOă 	íĺRÖA­*

global_step/secúłAÉůšj       Aď	Ő-ĺRÖA­*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossđó@§N~¤&       sOă 	ňKRÖA*

global_step/secv°AYA       Aď	> LRÖA*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossĆň@ÜoMŇ&       sOă 	żRÖAő*

global_step/secí÷ApŹť       Aď	żżRÖAő*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss`Â@Ü"C&       sOă 	tMCRÖAŮ*

global_step/sec!ßA ÄB       Aď	ô[CRÖAŮ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss}@\t&       sOă 	šcźRÖA˝*

global_step/secÇA:nű       Aď	rźRÖA˝*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossIá@îJ)k&       sOă 	VV*RÖAĄ*

global_step/sec éAż	Ë       Aď	ád*RÖAĄ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossl=@
c4ţ&       sOă 	bRÖA*

global_step/sec/pAů˙ł       Aď	RÖA*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss	@GfĐ§.       ĹËWú	)ĚRÖAč: output_dir/model1/model.ckptaU,X     <ČH	tmQRÖA"°

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
_output_shapes
: *
capacityč
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*9
_output_shapes'
%:::	
*
component_types
2	
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*'
_output_shapes
: *
data_formatNHWC*
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:@*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*'
_output_shapes
:@*
data_formatNHWC*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b( *
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read* 
_output_shapes
:
*
data_formatNHWC*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	
*
use_locking(*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_a( *
_output_shapes
:	
*
transpose_b( *
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
_output_shapes
:	
*
data_formatNHWC*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
_output_shapes
:*

axis *
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*-
_class#
!loc:@OptimizeLoss/learning_rate*
T0*
_output_shapes
: *
use_locking(*
validate_shape(

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*
_output_shapes
:	*

Tdim0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:
*
data_formatNHWC*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	
*
transpose_b( *
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_a( * 
_output_shapes
:
*
transpose_b(*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
transpose_b( *
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
: *
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: @*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*'
_output_shapes
:*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
: *
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
use_locking( *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
_output_shapes
: *
use_locking( *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
use_locking( *
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
use_locking( *
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel* 
_output_shapes
:
*
use_locking( *
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
_output_shapes	
:*
use_locking( *
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
use_locking( *
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
_output_shapes
: *
use_locking( *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
_output_shapes
: *
use_locking( *
T0
Ś
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
ů
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ą
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
§
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
ż
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11*
N*
_output_shapes
:*

axis *
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ť
$report_uninitialized_variables/ConstConst*Ň
valueČBĹBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/count*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ů
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

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ő
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
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
Ż
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
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0


1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ź
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ą
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ł
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ť
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ą
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Š
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
Á
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
Ş
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


&report_uninitialized_variables_1/ConstConst*˛
value¨BĽ
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

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ă
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

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ű
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

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ë
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
ł
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
ł
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ń
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
á
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0


3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
Ť
Merge/MergeSummaryMergeSummarySenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_99feda4f680e4068bc39a772a64578ff/part*
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
˙
save/SaveV2/tensor_namesConst*˛
value¨BĽ
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


save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0

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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
´
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_5Assign
dense/biassave/RestoreV2_5*
_class
loc:@dense/bias*
T0*
_output_shapes	
:*
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
 
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
_class
loc:@global_step*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
¸
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"sč)öx     řŻvĘ	YRÖAJéń
0ř/
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
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
Č
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
î
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
í
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

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype

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
ref"dtype
is_initialized
"
dtypetype
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
Ĺ
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
ë
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
2	
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
2	
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

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
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
&
QueueSizeV2

handle
size
ř
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint˙˙˙˙˙˙˙˙˙"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring 
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
ö
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 

Where	
input
	
index	*	1.2.0-rc12v1.2.0-rc0-24-g94484aa°

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

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
˛
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

"enqueue_input/random_shuffle_queueRandomShuffleQueueV2"/device:CPU:0*
	container *
shared_name *
seed2 *
component_types
2	*"
shapes
: ::
*

seed *
min_after_dequeueú*
capacityč*
_output_shapes
: 
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
˙
.enqueue_input/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_3Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_4Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_5Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_1QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_3enqueue_input/Placeholder_4enqueue_input/Placeholder_5"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_6Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_7Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_8Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_2QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_6enqueue_input/Placeholder_7enqueue_input/Placeholder_8"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	
o
enqueue_input/Placeholder_9Placeholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
p
enqueue_input/Placeholder_10Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
p
enqueue_input/Placeholder_11Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:

0enqueue_input/random_shuffle_queue_EnqueueMany_3QueueEnqueueManyV2"enqueue_input/random_shuffle_queueenqueue_input/Placeholder_9enqueue_input/Placeholder_10enqueue_input/Placeholder_11"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2	

(enqueue_input/random_shuffle_queue_CloseQueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues( 

*enqueue_input/random_shuffle_queue_Close_1QueueCloseV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
cancel_pending_enqueues(

'enqueue_input/random_shuffle_queue_SizeQueueSizeV2"enqueue_input/random_shuffle_queue"/device:CPU:0*
_output_shapes
: 
e
enqueue_input/sub/yConst"/device:CPU:0*
value
B :ú*
dtype0*
_output_shapes
: 

enqueue_input/subSub'enqueue_input/random_shuffle_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*

DstT0*
_output_shapes
: 
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *>ĂŽ:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
_output_shapes
: *
T0
ű
Xenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsConst"/device:CPU:0*d
value[BY BSenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full*
dtype0*
_output_shapes
: 

Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullScalarSummaryXenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full/tagsenqueue_input/mul"/device:CPU:0*
_output_shapes
: *
T0
t
"random_shuffle_queue_DequeueMany/nConst"/device:CPU:0*
value
B :*
dtype0*
_output_shapes
: 
˙
 random_shuffle_queue_DequeueManyQueueDequeueManyV2"enqueue_input/random_shuffle_queue"random_shuffle_queue_DequeueMany/n"/device:CPU:0*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2	*9
_output_shapes'
%:::	

Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§Ž˝* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§Ž=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
seed2 *

seed * 
_class
loc:@conv2d/kernel*
dtype0*
T0
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0
ł
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
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
T0

conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 

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
ś
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
Ţ
conv2d/convolutionConv2D"random_shuffle_queue_DequeueMany:1conv2d/kernel/read*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
data_formatNHWC*'
_output_shapes
: *
T0
U
conv2d/ReluReluconv2d/BiasAdd*'
_output_shapes
: *
T0
˛
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚL˝*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚL=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*
dtype0*
T0
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
ˇ
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
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*&
_output_shapes
: @*
T0

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0

conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@

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
ž
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
Ő
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*'
_output_shapes
:@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
data_formatNHWC*'
_output_shapes
:@*
T0
Y
conv2d_2/ReluReluconv2d_2/BiasAdd*'
_output_shapes
:@*
T0
ś
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
f
flatten/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
Ľ
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
˙˙˙˙˙˙˙˙˙*
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
{
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/stack*
Tshape0* 
_output_shapes
:
*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×ł]˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×ł]=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
seed2 *

seed *
_class
loc:@dense/kernel*
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0
Ľ
dense/kernel
VariableV2*
	container *
shape:
*
shared_name * 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
*
T0
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel* 
_output_shapes
:
*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:


dense/bias
VariableV2*
	container *
shape:*
shared_name *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:*
T0
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
transpose_b( *
transpose_a( * 
_output_shapes
:
*
T0
y
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
L

dense/ReluReludense/BiasAdd* 
_output_shapes
:
*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *č˝*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
dtype0*
T0
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0
§
dense_1/kernel
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*!
_class
loc:@dense_1/kernel*
dtype0
Đ
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	
*
T0
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
*
T0

dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:


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
ş
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

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *
_output_shapes
:	
*
T0
~
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*
_output_shapes
:	
*
T0
U
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
_output_shapes
:	
*
T0

softmax_cross_entropy_loss/CastCast"random_shuffle_queue_DequeueMany:2*

SrcT0*

DstT0*
_output_shapes
:	

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
 softmax_cross_entropy_loss/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

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
Î
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
_output_shapes
:*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ý
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0

"softmax_cross_entropy_loss/ReshapeReshapedense_2/Softmax!softmax_cross_entropy_loss/concat*
Tshape0*
_output_shapes
:	
*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
s
"softmax_cross_entropy_loss/Shape_2Const*
valueB"   
   *
dtype0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0

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
Ô
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
_output_shapes
:*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ĺ
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
­
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/Cast#softmax_cross_entropy_loss/concat_1*
Tshape0*
_output_shapes
:	
*
T0
ż
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*&
_output_shapes
::	
*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

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

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*

axis *
_output_shapes
:*
T0
Ű
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*
_output_shapes	
:*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

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

<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
š
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ó
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
Í
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:*
T0
Ř
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes	
:*
T0
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
ˇ
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
ľ
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
ť
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
˝
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
ź
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
¸
%softmax_cross_entropy_loss/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
ś
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
 
,OptimizeLoss/learning_rate/Initializer/ConstConst*
valueB
 *
×#<*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
­
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
î
!OptimizeLoss/learning_rate/AssignAssignOptimizeLoss/learning_rate,OptimizeLoss/learning_rate/Initializer/Const*
validate_shape(*-
_class#
!loc:@OptimizeLoss/learning_rate*
use_locking(*
_output_shapes
: *
T0

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
 *  ?*
dtype0*
_output_shapes
: 

OptimizeLoss/gradients/FillFillOptimizeLoss/gradients/ShapeOptimizeLoss/gradients/Const*
_output_shapes
: *
T0

GOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ř
COptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/GreaterOptimizeLoss/gradients/FillGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
ú
EOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/GreaterGOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/zeros_likeOptimizeLoss/gradients/Fill*
_output_shapes
: *
T0
ă
MOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOpD^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectF^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentityCOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/SelectN^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*V
_classL
JHloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
ő
WOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1N^OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
č
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/SumSumBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDivPOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ţ
BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ó
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
Ů
DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDivDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulMulUOptimizeLoss/gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum>OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/mulROptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
é
SOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

JOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyJOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0

KOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 

AOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTileDOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeKOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
_output_shapes
: *
T0

HOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshapeAOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_1_grad/TileHOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

IOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/TileTileBOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeIOptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ź
POptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ShapeBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ô
>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulMul?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:*
T0

>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/SumSum>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mulPOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape>OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
Ô
@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_2?OptimizeLoss/gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
_output_shapes	
:*
T0

@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/mul_1ROptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1BOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ß
KOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeE^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
î
SOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeL^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
_output_shapes	
:*
T0
ď
UOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityDOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1L^OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
¤
HOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeSOptimizeLoss/gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
x
!OptimizeLoss/gradients/zeros_likeConst*
valueB	
*    *
dtype0*
_output_shapes
:	


NOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
JOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsHOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeNOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*
_output_shapes
:	*
T0
ç
COptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulMulJOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
_output_shapes
:	
*
T0

DOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:

FOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshapeCOptimizeLoss/gradients/softmax_cross_entropy_loss/xentropy_grad/mulDOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	
*
T0
š
/OptimizeLoss/gradients/dense_2/Softmax_grad/mulMulFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapedense_2/Softmax*
_output_shapes
:	
*
T0

AOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
/OptimizeLoss/gradients/dense_2/Softmax_grad/SumSum/OptimizeLoss/gradients/dense_2/Softmax_grad/mulAOptimizeLoss/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes	
:*
T0

9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
â
3OptimizeLoss/gradients/dense_2/Softmax_grad/ReshapeReshape/OptimizeLoss/gradients/dense_2/Softmax_grad/Sum9OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape/shape*
Tshape0*
_output_shapes
:	*
T0
Ý
/OptimizeLoss/gradients/dense_2/Softmax_grad/subSubFOptimizeLoss/gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape3OptimizeLoss/gradients/dense_2/Softmax_grad/Reshape*
_output_shapes
:	
*
T0
¤
1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1Mul/OptimizeLoss/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
_output_shapes
:	
*
T0
ľ
7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:
*
T0
˛
<OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/Softmax_grad/mul_18^OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad
˛
DOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	
*
T0
ť
FOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad=^OptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@OptimizeLoss/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
ç
1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMulMatMulDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
ß
3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/ReluDOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	
*
T0
­
;OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_depsNoOp2^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul4^OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1
ą
COptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity1OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
ś
EOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity3OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1<^OptimizeLoss/gradients/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@OptimizeLoss/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
ˇ
/OptimizeLoss/gradients/dense/Relu_grad/ReluGradReluGradCOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu* 
_output_shapes
:
*
T0
˛
5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ź
:OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/Relu_grad/ReluGrad6^OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad
Ť
BOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/Relu_grad/ReluGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/Relu_grad/ReluGrad* 
_output_shapes
:
*
T0
´
DOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity5OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad;^OptimizeLoss/gradients/dense/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@OptimizeLoss/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
á
/OptimizeLoss/gradients/dense/MatMul_grad/MatMulMatMulBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
transpose_a( * 
_output_shapes
:
*
T0
á
1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeBOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(* 
_output_shapes
:
*
T0
§
9OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_depsNoOp0^OptimizeLoss/gradients/dense/MatMul_grad/MatMul2^OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1
Š
AOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity/OptimizeLoss/gradients/dense/MatMul_grad/MatMul:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*B
_class8
64loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul* 
_output_shapes
:
*
T0
Ż
COptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity1OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1:^OptimizeLoss/gradients/dense/MatMul_grad/tuple/group_deps*D
_class:
86loc:@OptimizeLoss/gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

1OptimizeLoss/gradients/flatten/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ô
3OptimizeLoss/gradients/flatten/Reshape_grad/ReshapeReshapeAOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency1OptimizeLoss/gradients/flatten/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:@*
T0
°
?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool3OptimizeLoss/gradients/flatten/Reshape_grad/Reshape*
ksize
*'
_output_shapes
:@*
data_formatNHWC*
paddingVALID*
strides
*
T0
Ŕ
2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGradReluGrad?OptimizeLoss/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*'
_output_shapes
:@*
T0
ˇ
8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
ľ
=OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp3^OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad9^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
ž
EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity2OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@OptimizeLoss/gradients/conv2d_2/Relu_grad/ReluGrad*'
_output_shapes
:@*
T0
ż
GOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad>^OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@OptimizeLoss/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0

6OptimizeLoss/gradients/conv2d_2/convolution_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
ú
DOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6OptimizeLoss/gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/readEOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
ţ
EOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool8OptimizeLoss/gradients/conv2d_2/convolution_grad/Shape_1EOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ř
AOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_depsNoOpE^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputF^OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
ę
IOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*'
_output_shapes
: *
T0
í
KOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEOptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterB^OptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@OptimizeLoss/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Ŕ
=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPoolIOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency*
ksize
*'
_output_shapes
: *
data_formatNHWC*
paddingVALID*
strides
*
T0
ş
0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGradReluGrad=OptimizeLoss/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*'
_output_shapes
: *
T0
ł
6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ż
;OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad7^OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad
ś
COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity0OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@OptimizeLoss/gradients/conv2d/Relu_grad/ReluGrad*'
_output_shapes
: *
T0
ˇ
EOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity6OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad<^OptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@OptimizeLoss/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0

4OptimizeLoss/gradients/conv2d/convolution_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ň
BOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4OptimizeLoss/gradients/conv2d/convolution_grad/Shapeconv2d/kernel/readCOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*'
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0

6OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:

COptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter"random_shuffle_queue_DequeueMany:16OptimizeLoss/gradients/conv2d/convolution_grad/Shape_1COptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
T0
Ň
?OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_depsNoOpC^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInputD^OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropFilter
â
GOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependencyIdentityBOptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput@^OptimizeLoss/gradients/conv2d/convolution_grad/tuple/group_deps*U
_classK
IGloc:@OptimizeLoss/gradients/conv2d/convolution_grad/Conv2DBackpropInput*'
_output_shapes
:*
T0
ĺ
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
ľ
<OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelOptimizeLoss/learning_rate/readIOptimizeLoss/gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*
use_locking( *&
_output_shapes
: *
T0

:OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
use_locking( *
_output_shapes
: *
T0
˝
>OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelOptimizeLoss/learning_rate/readKOptimizeLoss/gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*
use_locking( *&
_output_shapes
: @*
T0
§
<OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasOptimizeLoss/learning_rate/readGOptimizeLoss/gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
use_locking( *
_output_shapes
:@*
T0
Ś
;OptimizeLoss/train/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelOptimizeLoss/learning_rate/readCOptimizeLoss/gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
use_locking( * 
_output_shapes
:
*
T0

9OptimizeLoss/train/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasOptimizeLoss/learning_rate/readDOptimizeLoss/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_locking( *
_output_shapes	
:*
T0
­
=OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelOptimizeLoss/learning_rate/readEOptimizeLoss/gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
use_locking( *
_output_shapes
:	
*
T0
Ł
;OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasOptimizeLoss/learning_rate/readFOptimizeLoss/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
use_locking( *
_output_shapes
:
*
T0

OptimizeLoss/train/updateNoOp=^OptimizeLoss/train/update_conv2d/kernel/ApplyGradientDescent;^OptimizeLoss/train/update_conv2d/bias/ApplyGradientDescent?^OptimizeLoss/train/update_conv2d_1/kernel/ApplyGradientDescent=^OptimizeLoss/train/update_conv2d_1/bias/ApplyGradientDescent<^OptimizeLoss/train/update_dense/kernel/ApplyGradientDescent:^OptimizeLoss/train/update_dense/bias/ApplyGradientDescent>^OptimizeLoss/train/update_dense_1/kernel/ApplyGradientDescent<^OptimizeLoss/train/update_dense_1/bias/ApplyGradientDescent

OptimizeLoss/train/valueConst^OptimizeLoss/train/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

OptimizeLoss/train	AssignAddglobal_stepOptimizeLoss/train/value*
_class
loc:@global_step*
use_locking( *
_output_shapes
: *
T0	
¸
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
e
ArgMaxArgMaxdense_2/SoftmaxArgMax/dimension*

Tidx0*
_output_shapes	
:*
T0
M
SoftmaxSoftmaxdense_2/Softmax*
_output_shapes
:	
*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
i
ArgMax_1ArgMaxdense_2/SoftmaxArgMax_1/dimension*

Tidx0*
_output_shapes	
:*
T0
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
|
ArgMax_2ArgMax"random_shuffle_queue_DequeueMany:2ArgMax_2/dimension*

Tidx0*
_output_shapes	
:*
T0
H
EqualEqualArgMax_2ArgMax_1*
_output_shapes	
:*
T0	
K
ToFloatCastEqual*

SrcT0
*

DstT0*
_output_shapes	
:
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
Ź
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
Ž
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
P
accuracy/SizeConst*
value
B :*
dtype0*
_output_shapes
: 
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

accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
_output_shapes
: *
T0
Ś
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
ů
initNoOp^global_step/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign"^OptimizeLoss/learning_rate/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ą
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ľ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
§
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ł
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
ż
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedaccuracy/total*!
_class
loc:@accuracy/total*
dtype0*
_output_shapes
: 
¨
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedaccuracy/count*!
_class
loc:@accuracy/count*
dtype0*
_output_shapes
: 

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11*
N*

axis *
_output_shapes
:*
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ť
$report_uninitialized_variables/ConstConst*Ň
valueČBĹBglobal_stepBconv2d/kernelBconv2d/biasBconv2d_1/kernelBconv2d_1/biasBdense/kernelB
dense/biasBdense_1/kernelBdense_1/biasBOptimizeLoss/learning_rateBaccuracy/totalBaccuracy/count*
dtype0*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ů
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

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ő
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
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
Ż
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
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0


1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ź
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ą
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ł
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Ť
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
§
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ą
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Š
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ľ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
Á
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedOptimizeLoss/learning_rate*-
_class#
!loc:@OptimizeLoss/learning_rate*
dtype0*
_output_shapes
: 
Ş
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


&report_uninitialized_variables_1/ConstConst*˛
value¨BĽ
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

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ă
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

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ű
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

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ë
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
ł
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
ł
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ń
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
á
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0


3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
init_2NoOp^accuracy/total/Assign^accuracy/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
Ť
Merge/MergeSummaryMergeSummarySenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_fullOptimizeLoss/loss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_99feda4f680e4068bc39a772a64578ff/part*
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
˙
save/SaveV2/tensor_namesConst*˛
value¨BĽ
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


save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizeLoss/learning_rateconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2
	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0

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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
´
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_5Assign
dense/biassave/RestoreV2_5*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:*
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_6Assigndense/kernelsave/RestoreV2_6*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(* 
_output_shapes
:
*
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_8Assigndense_1/kernelsave/RestoreV2_8*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2	*
_output_shapes
:
 
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
validate_shape(*
_class
loc:@global_step*
use_locking(*
_output_shapes
: *
T0	
¸
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
ready_op


concat:0" 
global_step

global_step:0"
init_op


group_deps"Ü
queue_runnersĘÇ
Ä
"enqueue_input/random_shuffle_queue.enqueue_input/random_shuffle_queue_EnqueueMany0enqueue_input/random_shuffle_queue_EnqueueMany_10enqueue_input/random_shuffle_queue_EnqueueMany_20enqueue_input/random_shuffle_queue_EnqueueMany_3(enqueue_input/random_shuffle_queue_Close"*enqueue_input/random_shuffle_queue_Close_1*"9
local_variables&
$
accuracy/total:0
accuracy/count:0"
	variables
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
OptimizeLoss/learning_rate:0!OptimizeLoss/learning_rate/Assign!OptimizeLoss/learning_rate/read:0"T
lossesJ
H
"softmax_cross_entropy_loss/value:0
"softmax_cross_entropy_loss/value:0"&

summary_op

Merge/MergeSummary:0"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
trainable_variablesďě
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
save/Const:0save/Identity:0save/restore_all (5 @F8"{
	summariesn
l
Uenqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full:0
OptimizeLoss/loss:0"!
local_init_op

group_deps_1óTgL.       ĹËWú	ÍtRÖAé: output_dir/model1/model.ckptŐ\.       űţ	OtRÖAé:WZô       Aď	0tRÖAé*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss˘@)|&       sOă 	ýsRÖAÍ*

global_step/secjHAl       Aď	tRÖAÍ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossţČń?ń˘d&       sOă 	FŞäRÖAą	*

global_step/secößAQY|       Aď	Á¸äRÖAą	*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossŃâ?RŔë&       sOă 	s÷_RÖA
*

global_step/sec>üA-mĆ       Aď	Ł`RÖA
*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossggŕ?Nęčç&       sOă 	DÍÝRÖAů
*

global_step/secŮAŞ8W       Aď	ÝÝRÖAů
*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossśzß?ˇk&       sOă 	]RÖAÝ*

global_step/sec|mAĐý3$       Aď	Š]RÖAÝ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss¨2ŕ?1îú&       sOă 	ß äRÖAÁ*

global_step/secśA*="       Aď	'/äRÖAÁ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss]ĹŐ?Ť˛Ž&&       sOă 	Ú
RĄRÖAĽ*

global_step/secaěAK*T       Aď	żRĄRÖAĽ*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss',Ů?y=f4&       sOă 	Ö¸ż˘RÖA*

global_step/secfAM3Ę       Aď	ŞÉż˘RÖA*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/loss˘Ő?Ťť|7&       sOă 	^4¤RÖAí*

global_step/secQPAą\s       Aď	ÂĽ4¤RÖAí*v
Z
Senqueue_input/queue/enqueue_input/random_shuffle_queuefraction_over_250_of_750_full  ?

OptimizeLoss/lossś\Ő?ăęr.       ĹËWú	F˘´ĽRÖAĐ: output_dir/model1/model.ckptRř