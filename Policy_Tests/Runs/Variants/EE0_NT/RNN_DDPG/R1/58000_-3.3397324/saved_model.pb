��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
ActorRnnNetwork/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*.
shared_nameActorRnnNetwork/action/kernel
�
1ActorRnnNetwork/action/kernel/Read/ReadVariableOpReadVariableOpActorRnnNetwork/action/kernel*
_output_shapes

:2*
dtype0
�
ActorRnnNetwork/action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameActorRnnNetwork/action/bias
�
/ActorRnnNetwork/action/bias/Read/ReadVariableOpReadVariableOpActorRnnNetwork/action/bias*
_output_shapes
:*
dtype0
�
%ActorRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*6
shared_name'%ActorRnnNetwork/dynamic_unroll/kernel
�
9ActorRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOp%ActorRnnNetwork/dynamic_unroll/kernel* 
_output_shapes
:
��*
dtype0
�
/ActorRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(�*@
shared_name1/ActorRnnNetwork/dynamic_unroll/recurrent_kernel
�
CActorRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOp/ActorRnnNetwork/dynamic_unroll/recurrent_kernel*
_output_shapes
:	(�*
dtype0
�
#ActorRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#ActorRnnNetwork/dynamic_unroll/bias
�
7ActorRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOp#ActorRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:�*
dtype0
�
&ActorRnnNetwork/input_mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*7
shared_name(&ActorRnnNetwork/input_mlp/dense/kernel
�
:ActorRnnNetwork/input_mlp/dense/kernel/Read/ReadVariableOpReadVariableOp&ActorRnnNetwork/input_mlp/dense/kernel*
_output_shapes
:	�*
dtype0
�
$ActorRnnNetwork/input_mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$ActorRnnNetwork/input_mlp/dense/bias
�
8ActorRnnNetwork/input_mlp/dense/bias/Read/ReadVariableOpReadVariableOp$ActorRnnNetwork/input_mlp/dense/bias*
_output_shapes	
:�*
dtype0
�
(ActorRnnNetwork/input_mlp/dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(ActorRnnNetwork/input_mlp/dense/kernel_1
�
<ActorRnnNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOpReadVariableOp(ActorRnnNetwork/input_mlp/dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&ActorRnnNetwork/input_mlp/dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&ActorRnnNetwork/input_mlp/dense/bias_1
�
:ActorRnnNetwork/input_mlp/dense/bias_1/Read/ReadVariableOpReadVariableOp&ActorRnnNetwork/input_mlp/dense/bias_1*
_output_shapes	
:�*
dtype0
�
#ActorRnnNetwork/output/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(2*4
shared_name%#ActorRnnNetwork/output/dense/kernel
�
7ActorRnnNetwork/output/dense/kernel/Read/ReadVariableOpReadVariableOp#ActorRnnNetwork/output/dense/kernel*
_output_shapes

:(2*
dtype0
�
!ActorRnnNetwork/output/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!ActorRnnNetwork/output/dense/bias
�
5ActorRnnNetwork/output/dense/bias/Read/ReadVariableOpReadVariableOp!ActorRnnNetwork/output/dense/bias*
_output_shapes
:2*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�$
k
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures
 
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
	2

3
4
5
6
7
8
9
10

0
1
2
 
_]
VARIABLE_VALUEActorRnnNetwork/action/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEActorRnnNetwork/action/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%ActorRnnNetwork/dynamic_unroll/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE/ActorRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#ActorRnnNetwork/dynamic_unroll/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&ActorRnnNetwork/input_mlp/dense/kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE$ActorRnnNetwork/input_mlp/dense/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(ActorRnnNetwork/input_mlp/dense/kernel_1,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&ActorRnnNetwork/input_mlp/dense/bias_1,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#ActorRnnNetwork/output/dense/kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE!ActorRnnNetwork/output/dense/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

ref
1
�
_state_spec
_flat_action_spec
_input_layers
_dynamic_unroll
_output_layers
_action_layers
regularization_losses
trainable_variables
 	variables
!	keras_api
 

	state
1
 
 

"0
#1
$2
\
%cell
&regularization_losses
'trainable_variables
(	variables
)	keras_api

*0
+1

,0
 
N
0
1
2
3
	4

5
6
7
8
9
10
N
0
1
2
3
	4

5
6
7
8
9
10
�
regularization_losses
-metrics
.non_trainable_variables
/layer_metrics
0layer_regularization_losses
trainable_variables

1layers
 	variables
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

kernel
bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

kernel
bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
�
>
state_size

	kernel

recurrent_kernel
bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
 

	0

1
2

	0

1
2
�
&regularization_losses
Cmetrics
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses
'trainable_variables

Glayers
(	variables
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

kernel
bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

kernel
bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
 
 
 
 
1
"0
#1
$2
3
*4
+5
,6
 
 
 
�
2regularization_losses
Tmetrics
Unon_trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
3trainable_variables

Xlayers
4	variables
 

0
1

0
1
�
6regularization_losses
Ymetrics
Znon_trainable_variables
[layer_metrics
\layer_regularization_losses
7trainable_variables

]layers
8	variables
 

0
1

0
1
�
:regularization_losses
^metrics
_non_trainable_variables
`layer_metrics
alayer_regularization_losses
;trainable_variables

blayers
<	variables
 
 

	0

1
2

	0

1
2
�
?regularization_losses
cmetrics
dnon_trainable_variables
elayer_metrics
flayer_regularization_losses
@trainable_variables

glayers
A	variables
 
 
 
 

%0
 
 
 
�
Hregularization_losses
hmetrics
inon_trainable_variables
jlayer_metrics
klayer_regularization_losses
Itrainable_variables

llayers
J	variables
 

0
1

0
1
�
Lregularization_losses
mmetrics
nnon_trainable_variables
olayer_metrics
player_regularization_losses
Mtrainable_variables

qlayers
N	variables
 

0
1

0
1
�
Pregularization_losses
rmetrics
snon_trainable_variables
tlayer_metrics
ulayer_regularization_losses
Qtrainable_variables

vlayers
R	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0/observationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
action_0/rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0/step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m

action_1/0Placeholder*'
_output_shapes
:���������(*
dtype0*
shape:���������(
m

action_1/1Placeholder*'
_output_shapes
:���������(*
dtype0*
shape:���������(
�
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type
action_1/0
action_1/1&ActorRnnNetwork/input_mlp/dense/kernel$ActorRnnNetwork/input_mlp/dense/bias(ActorRnnNetwork/input_mlp/dense/kernel_1&ActorRnnNetwork/input_mlp/dense/bias_1%ActorRnnNetwork/dynamic_unroll/kernel/ActorRnnNetwork/dynamic_unroll/recurrent_kernel#ActorRnnNetwork/dynamic_unroll/bias#ActorRnnNetwork/output/dense/kernel!ActorRnnNetwork/output/dense/biasActorRnnNetwork/action/kernelActorRnnNetwork/action/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������(:���������(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6872139
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6872148
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6872160
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6872156
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp1ActorRnnNetwork/action/kernel/Read/ReadVariableOp/ActorRnnNetwork/action/bias/Read/ReadVariableOp9ActorRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpCActorRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOp7ActorRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOp:ActorRnnNetwork/input_mlp/dense/kernel/Read/ReadVariableOp8ActorRnnNetwork/input_mlp/dense/bias/Read/ReadVariableOp<ActorRnnNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOp:ActorRnnNetwork/input_mlp/dense/bias_1/Read/ReadVariableOp7ActorRnnNetwork/output/dense/kernel/Read/ReadVariableOp5ActorRnnNetwork/output/dense/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_6872230
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableActorRnnNetwork/action/kernelActorRnnNetwork/action/bias%ActorRnnNetwork/dynamic_unroll/kernel/ActorRnnNetwork/dynamic_unroll/recurrent_kernel#ActorRnnNetwork/dynamic_unroll/bias&ActorRnnNetwork/input_mlp/dense/kernel$ActorRnnNetwork/input_mlp/dense/bias(ActorRnnNetwork/input_mlp/dense/kernel_1&ActorRnnNetwork/input_mlp/dense/bias_1#ActorRnnNetwork/output/dense/kernel!ActorRnnNetwork/output/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_6872276��
ͻ
�
__inference_action_3405345
	step_type

reward
discount
observation
unknown
	unknown_0Q
>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource:	�N
?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource:	�T
@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��P
Aactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�[
Gactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
��\
Iactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(�W
Hactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	�M
;actorrnnnetwork_output_dense_matmul_readvariableop_resource:(2J
<actorrnnnetwork_output_dense_biasadd_readvariableop_resource:2G
5actorrnnnetwork_action_matmul_readvariableop_resource:2D
6actorrnnnetwork_action_biasadd_readvariableop_resource:
identity

identity_1

identity_2��-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�,ActorRnnNetwork/action/MatMul/ReadVariableOp�?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpF
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:���������2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis�
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:���������2	
Reshape}
SelectV2SelectV2Reshape:output:0zeros:output:0unknown*
T0*'
_output_shapes
:���������(2

SelectV2�

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0	unknown_0*
T0*'
_output_shapes
:���������(2

SelectV2_1�
ActorRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
ActorRnnNetwork/ExpandDims/dim�
ActorRnnNetwork/ExpandDims
ExpandDimsobservation'ActorRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims�
 ActorRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 ActorRnnNetwork/ExpandDims_1/dim�
ActorRnnNetwork/ExpandDims_1
ExpandDims	step_type)ActorRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims_1�
#ActorRnnNetwork/batch_flatten/ShapeShape#ActorRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2%
#ActorRnnNetwork/batch_flatten/Shape�
+ActorRnnNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2-
+ActorRnnNetwork/batch_flatten/Reshape/shape�
%ActorRnnNetwork/batch_flatten/ReshapeReshape#ActorRnnNetwork/ExpandDims:output:04ActorRnnNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2'
%ActorRnnNetwork/batch_flatten/Reshape�
ActorRnnNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/flatten/Const�
ActorRnnNetwork/flatten/ReshapeReshape.ActorRnnNetwork/batch_flatten/Reshape:output:0&ActorRnnNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2!
ActorRnnNetwork/flatten/Reshape�
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype027
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�
&ActorRnnNetwork/input_mlp/dense/MatMulMatMul(ActorRnnNetwork/flatten/Reshape:output:0=ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/MatMul�
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�
'ActorRnnNetwork/input_mlp/dense/BiasAddBiasAdd0ActorRnnNetwork/input_mlp/dense/MatMul:product:0>ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'ActorRnnNetwork/input_mlp/dense/BiasAdd�
$ActorRnnNetwork/input_mlp/dense/ReluRelu0ActorRnnNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2&
$ActorRnnNetwork/input_mlp/dense/Relu�
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�
(ActorRnnNetwork/input_mlp/dense/MatMul_1MatMul2ActorRnnNetwork/input_mlp/dense/Relu:activations:0?ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(ActorRnnNetwork/input_mlp/dense/MatMul_1�
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOpAactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1BiasAdd2ActorRnnNetwork/input_mlp/dense/MatMul_1:product:0@ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1�
&ActorRnnNetwork/input_mlp/dense/Relu_1Relu2ActorRnnNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/Relu_1�
3ActorRnnNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3ActorRnnNetwork/batch_unflatten/strided_slice/stack�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2�
-ActorRnnNetwork/batch_unflatten/strided_sliceStridedSlice,ActorRnnNetwork/batch_flatten/Shape:output:0<ActorRnnNetwork/batch_unflatten/strided_slice/stack:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_1:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-ActorRnnNetwork/batch_unflatten/strided_slice�
%ActorRnnNetwork/batch_unflatten/ShapeShape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_unflatten/Shape�
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stack�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2�
/ActorRnnNetwork/batch_unflatten/strided_slice_1StridedSlice.ActorRnnNetwork/batch_unflatten/Shape:output:0>ActorRnnNetwork/batch_unflatten/strided_slice_1/stack:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask21
/ActorRnnNetwork/batch_unflatten/strided_slice_1�
+ActorRnnNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+ActorRnnNetwork/batch_unflatten/concat/axis�
&ActorRnnNetwork/batch_unflatten/concatConcatV26ActorRnnNetwork/batch_unflatten/strided_slice:output:08ActorRnnNetwork/batch_unflatten/strided_slice_1:output:04ActorRnnNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2(
&ActorRnnNetwork/batch_unflatten/concat�
'ActorRnnNetwork/batch_unflatten/ReshapeReshape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0/ActorRnnNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:����������2)
'ActorRnnNetwork/batch_unflatten/Reshape�
"ActorRnnNetwork/reset_mask/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ActorRnnNetwork/reset_mask/Equal/y�
 ActorRnnNetwork/reset_mask/EqualEqual%ActorRnnNetwork/ExpandDims_1:output:0+ActorRnnNetwork/reset_mask/Equal/y:output:0*
T0*'
_output_shapes
:���������2"
 ActorRnnNetwork/reset_mask/Equal�
#ActorRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#ActorRnnNetwork/dynamic_unroll/Rank�
*ActorRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/start�
*ActorRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/delta�
$ActorRnnNetwork/dynamic_unroll/rangeRange3ActorRnnNetwork/dynamic_unroll/range/start:output:0,ActorRnnNetwork/dynamic_unroll/Rank:output:03ActorRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/range�
.ActorRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       20
.ActorRnnNetwork/dynamic_unroll/concat/values_0�
*ActorRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ActorRnnNetwork/dynamic_unroll/concat/axis�
%ActorRnnNetwork/dynamic_unroll/concatConcatV27ActorRnnNetwork/dynamic_unroll/concat/values_0:output:0-ActorRnnNetwork/dynamic_unroll/range:output:03ActorRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%ActorRnnNetwork/dynamic_unroll/concat�
(ActorRnnNetwork/dynamic_unroll/transpose	Transpose0ActorRnnNetwork/batch_unflatten/Reshape:output:0.ActorRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:����������2*
(ActorRnnNetwork/dynamic_unroll/transpose�
$ActorRnnNetwork/dynamic_unroll/ShapeShape,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/Shape�
2ActorRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2ActorRnnNetwork/dynamic_unroll/strided_slice/stack�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2�
,ActorRnnNetwork/dynamic_unroll/strided_sliceStridedSlice-ActorRnnNetwork/dynamic_unroll/Shape:output:0;ActorRnnNetwork/dynamic_unroll/strided_slice/stack:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,ActorRnnNetwork/dynamic_unroll/strided_slice�
/ActorRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/ActorRnnNetwork/dynamic_unroll/transpose_1/perm�
*ActorRnnNetwork/dynamic_unroll/transpose_1	Transpose$ActorRnnNetwork/reset_mask/Equal:z:08ActorRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:���������2,
*ActorRnnNetwork/dynamic_unroll/transpose_1�
*ActorRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2,
*ActorRnnNetwork/dynamic_unroll/zeros/mul/y�
(ActorRnnNetwork/dynamic_unroll/zeros/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:03ActorRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(ActorRnnNetwork/dynamic_unroll/zeros/mul�
+ActorRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2-
+ActorRnnNetwork/dynamic_unroll/zeros/Less/y�
)ActorRnnNetwork/dynamic_unroll/zeros/LessLess,ActorRnnNetwork/dynamic_unroll/zeros/mul:z:04ActorRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)ActorRnnNetwork/dynamic_unroll/zeros/Less�
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2/
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1�
+ActorRnnNetwork/dynamic_unroll/zeros/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:06ActorRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+ActorRnnNetwork/dynamic_unroll/zeros/packed�
*ActorRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*ActorRnnNetwork/dynamic_unroll/zeros/Const�
$ActorRnnNetwork/dynamic_unroll/zerosFill4ActorRnnNetwork/dynamic_unroll/zeros/packed:output:03ActorRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:���������(2&
$ActorRnnNetwork/dynamic_unroll/zeros�
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y�
*ActorRnnNetwork/dynamic_unroll/zeros_1/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*ActorRnnNetwork/dynamic_unroll/zeros_1/mul�
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y�
+ActorRnnNetwork/dynamic_unroll/zeros_1/LessLess.ActorRnnNetwork/dynamic_unroll/zeros_1/mul:z:06ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+ActorRnnNetwork/dynamic_unroll/zeros_1/Less�
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(21
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1�
-ActorRnnNetwork/dynamic_unroll/zeros_1/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:08ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/packed�
,ActorRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/Const�
&ActorRnnNetwork/dynamic_unroll/zeros_1Fill6ActorRnnNetwork/dynamic_unroll/zeros_1/packed:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2(
&ActorRnnNetwork/dynamic_unroll/zeros_1�
&ActorRnnNetwork/dynamic_unroll/SqueezeSqueeze,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
 2(
&ActorRnnNetwork/dynamic_unroll/Squeeze�
(ActorRnnNetwork/dynamic_unroll/Squeeze_1Squeeze.ActorRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:���������*
squeeze_dims
 2*
(ActorRnnNetwork/dynamic_unroll/Squeeze_1�
%ActorRnnNetwork/dynamic_unroll/SelectSelect1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0-ActorRnnNetwork/dynamic_unroll/zeros:output:0SelectV2:output:0*
T0*'
_output_shapes
:���������(2'
%ActorRnnNetwork/dynamic_unroll/Select�
'ActorRnnNetwork/dynamic_unroll/Select_1Select1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0/ActorRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/dynamic_unroll/Select_1�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpGactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02@
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul/ActorRnnNetwork/dynamic_unroll/Squeeze:output:0FActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(�*
dtype02B
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul.ActorRnnNetwork/dynamic_unroll/Select:output:0HActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/addAddV29ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0;ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/add�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02A
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd0ActorRnnNetwork/dynamic_unroll/lstm_cell/add:z:0GActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd�
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/splitSplitAActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:09ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������(:���������(:���������(:���������(*
	num_split20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/split�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������(22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mulMul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:00ActorRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:���������(2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mul�
-ActorRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������(2/
-ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul4ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:01ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV20ActorRnnNetwork/dynamic_unroll/lstm_cell/mul:z:02ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������(21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:03ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2�
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dim�
)ActorRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:06ActorRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������(2+
)ActorRnnNetwork/dynamic_unroll/ExpandDims�
%ActorRnnNetwork/batch_flatten_1/ShapeShape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_flatten_1/Shape�
-ActorRnnNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����(   2/
-ActorRnnNetwork/batch_flatten_1/Reshape/shape�
'ActorRnnNetwork/batch_flatten_1/ReshapeReshape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:06ActorRnnNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/batch_flatten_1/Reshape�
ActorRnnNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����(   2!
ActorRnnNetwork/flatten_1/Const�
!ActorRnnNetwork/flatten_1/ReshapeReshape0ActorRnnNetwork/batch_flatten_1/Reshape:output:0(ActorRnnNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������(2#
!ActorRnnNetwork/flatten_1/Reshape�
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpReadVariableOp;actorrnnnetwork_output_dense_matmul_readvariableop_resource*
_output_shapes

:(2*
dtype024
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp�
#ActorRnnNetwork/output/dense/MatMulMatMul*ActorRnnNetwork/flatten_1/Reshape:output:0:ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22%
#ActorRnnNetwork/output/dense/MatMul�
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOpReadVariableOp<actorrnnnetwork_output_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype025
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�
$ActorRnnNetwork/output/dense/BiasAddBiasAdd-ActorRnnNetwork/output/dense/MatMul:product:0;ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22&
$ActorRnnNetwork/output/dense/BiasAdd�
!ActorRnnNetwork/output/dense/ReluRelu-ActorRnnNetwork/output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������22#
!ActorRnnNetwork/output/dense/Relu�
,ActorRnnNetwork/action/MatMul/ReadVariableOpReadVariableOp5actorrnnnetwork_action_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,ActorRnnNetwork/action/MatMul/ReadVariableOp�
ActorRnnNetwork/action/MatMulMatMul/ActorRnnNetwork/output/dense/Relu:activations:04ActorRnnNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/MatMul�
-ActorRnnNetwork/action/BiasAdd/ReadVariableOpReadVariableOp6actorrnnnetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�
ActorRnnNetwork/action/BiasAddBiasAdd'ActorRnnNetwork/action/MatMul:product:05ActorRnnNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
ActorRnnNetwork/action/BiasAdd�
ActorRnnNetwork/action/TanhTanh'ActorRnnNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/Tanh�
ActorRnnNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/Reshape/shape�
ActorRnnNetwork/ReshapeReshapeActorRnnNetwork/action/Tanh:y:0&ActorRnnNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/Reshapes
ActorRnnNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/mul/x�
ActorRnnNetwork/mulMulActorRnnNetwork/mul/x:output:0 ActorRnnNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/muls
ActorRnnNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/add/x�
ActorRnnNetwork/addAddV2ActorRnnNetwork/add/x:output:0ActorRnnNetwork/mul:z:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/add�
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stack�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2�
/ActorRnnNetwork/batch_unflatten_1/strided_sliceStridedSlice.ActorRnnNetwork/batch_flatten_1/Shape:output:0>ActorRnnNetwork/batch_unflatten_1/strided_slice/stack:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask21
/ActorRnnNetwork/batch_unflatten_1/strided_slice�
'ActorRnnNetwork/batch_unflatten_1/ShapeShapeActorRnnNetwork/add:z:0*
T0*
_output_shapes
:*
out_type0	2)
'ActorRnnNetwork/batch_unflatten_1/Shape�
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2�
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1StridedSlice0ActorRnnNetwork/batch_unflatten_1/Shape:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask23
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1�
-ActorRnnNetwork/batch_unflatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-ActorRnnNetwork/batch_unflatten_1/concat/axis�
(ActorRnnNetwork/batch_unflatten_1/concatConcatV28ActorRnnNetwork/batch_unflatten_1/strided_slice:output:0:ActorRnnNetwork/batch_unflatten_1/strided_slice_1:output:06ActorRnnNetwork/batch_unflatten_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:2*
(ActorRnnNetwork/batch_unflatten_1/concat�
)ActorRnnNetwork/batch_unflatten_1/ReshapeReshapeActorRnnNetwork/add:z:01ActorRnnNetwork/batch_unflatten_1/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:���������2+
)ActorRnnNetwork/batch_unflatten_1/Reshape�
ActorRnnNetwork/SqueezeSqueeze2ActorRnnNetwork/batch_unflatten_1/Reshape:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
2
ActorRnnNetwork/Squeezem
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShape ActorRnnNetwork/Squeeze:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastTo ActorRnnNetwork/Squeeze:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:���������2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_1�

Identity_2Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_2�
NoOpNoOp.^ActorRnnNetwork/action/BiasAdd/ReadVariableOp-^ActorRnnNetwork/action/MatMul/ReadVariableOp@^ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpA^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp7^ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp9^ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp8^ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4^ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3^ActorRnnNetwork/output/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 2^
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp-ActorRnnNetwork/action/BiasAdd/ReadVariableOp2\
,ActorRnnNetwork/action/MatMul/ReadVariableOp,ActorRnnNetwork/action/MatMul/ReadVariableOp2�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2p
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2t
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp2r
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2j
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp2h
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation:JF
'
_output_shapes
:���������(

_user_specified_name0:JF
'
_output_shapes
:���������(

_user_specified_name1
�
U
%__inference_signature_wrapper_6872148

batch_size
identity

identity_1�
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_34051062
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������(2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������(2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�9
�	
#__inference__traced_restore_6872276
file_prefix#
assignvariableop_variable:	 B
0assignvariableop_1_actorrnnnetwork_action_kernel:2<
.assignvariableop_2_actorrnnnetwork_action_bias:L
8assignvariableop_3_actorrnnnetwork_dynamic_unroll_kernel:
��U
Bassignvariableop_4_actorrnnnetwork_dynamic_unroll_recurrent_kernel:	(�E
6assignvariableop_5_actorrnnnetwork_dynamic_unroll_bias:	�L
9assignvariableop_6_actorrnnnetwork_input_mlp_dense_kernel:	�F
7assignvariableop_7_actorrnnnetwork_input_mlp_dense_bias:	�O
;assignvariableop_8_actorrnnnetwork_input_mlp_dense_kernel_1:
��H
9assignvariableop_9_actorrnnnetwork_input_mlp_dense_bias_1:	�I
7assignvariableop_10_actorrnnnetwork_output_dense_kernel:(2C
5assignvariableop_11_actorrnnnetwork_output_dense_bias:2
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_actorrnnnetwork_action_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_actorrnnnetwork_action_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_actorrnnnetwork_dynamic_unroll_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpBassignvariableop_4_actorrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_actorrnnnetwork_dynamic_unroll_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_actorrnnnetwork_input_mlp_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp7assignvariableop_7_actorrnnnetwork_input_mlp_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp;assignvariableop_8_actorrnnnetwork_input_mlp_dense_kernel_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_actorrnnnetwork_input_mlp_dense_bias_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_actorrnnnetwork_output_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_actorrnnnetwork_output_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_function_with_signature_3405044
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	(�
	unknown_7:	�
	unknown_8:(2
	unknown_9:2

unknown_10:2

unknown_11:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������(:���������(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_action_34050152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������(2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������(2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:LH
'
_output_shapes
:���������(

_user_specified_name1/0:LH
'
_output_shapes
:���������(

_user_specified_name1/1
�
k
+__inference_function_with_signature_3405122
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *$
fR
__inference_<lambda>_1616182
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
�
%__inference_signature_wrapper_6872139
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	(�
	unknown_7:	�
	unknown_8:(2
	unknown_9:2

unknown_10:2

unknown_11:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������(:���������(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_34050442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������(2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������(2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type:LH
'
_output_shapes
:���������(

_user_specified_name1/0:LH
'
_output_shapes
:���������(

_user_specified_name1/1
�
e
%__inference_signature_wrapper_6872156
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_34051222
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
��
�
#__inference_distribution_fn_3405734
	step_type

reward
discount
observation
unknown
	unknown_0Q
>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource:	�N
?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource:	�T
@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��P
Aactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�[
Gactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
��\
Iactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(�W
Hactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	�M
;actorrnnnetwork_output_dense_matmul_readvariableop_resource:(2J
<actorrnnnetwork_output_dense_biasadd_readvariableop_resource:2G
5actorrnnnetwork_action_matmul_readvariableop_resource:2D
6actorrnnnetwork_action_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4��-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�,ActorRnnNetwork/action/MatMul/ReadVariableOp�?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpF
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:���������2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis�
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:���������2	
Reshape}
SelectV2SelectV2Reshape:output:0zeros:output:0unknown*
T0*'
_output_shapes
:���������(2

SelectV2�

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0	unknown_0*
T0*'
_output_shapes
:���������(2

SelectV2_1�
ActorRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
ActorRnnNetwork/ExpandDims/dim�
ActorRnnNetwork/ExpandDims
ExpandDimsobservation'ActorRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims�
 ActorRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 ActorRnnNetwork/ExpandDims_1/dim�
ActorRnnNetwork/ExpandDims_1
ExpandDims	step_type)ActorRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims_1�
#ActorRnnNetwork/batch_flatten/ShapeShape#ActorRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2%
#ActorRnnNetwork/batch_flatten/Shape�
+ActorRnnNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2-
+ActorRnnNetwork/batch_flatten/Reshape/shape�
%ActorRnnNetwork/batch_flatten/ReshapeReshape#ActorRnnNetwork/ExpandDims:output:04ActorRnnNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2'
%ActorRnnNetwork/batch_flatten/Reshape�
ActorRnnNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/flatten/Const�
ActorRnnNetwork/flatten/ReshapeReshape.ActorRnnNetwork/batch_flatten/Reshape:output:0&ActorRnnNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2!
ActorRnnNetwork/flatten/Reshape�
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype027
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�
&ActorRnnNetwork/input_mlp/dense/MatMulMatMul(ActorRnnNetwork/flatten/Reshape:output:0=ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/MatMul�
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�
'ActorRnnNetwork/input_mlp/dense/BiasAddBiasAdd0ActorRnnNetwork/input_mlp/dense/MatMul:product:0>ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'ActorRnnNetwork/input_mlp/dense/BiasAdd�
$ActorRnnNetwork/input_mlp/dense/ReluRelu0ActorRnnNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2&
$ActorRnnNetwork/input_mlp/dense/Relu�
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�
(ActorRnnNetwork/input_mlp/dense/MatMul_1MatMul2ActorRnnNetwork/input_mlp/dense/Relu:activations:0?ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(ActorRnnNetwork/input_mlp/dense/MatMul_1�
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOpAactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1BiasAdd2ActorRnnNetwork/input_mlp/dense/MatMul_1:product:0@ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1�
&ActorRnnNetwork/input_mlp/dense/Relu_1Relu2ActorRnnNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/Relu_1�
3ActorRnnNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3ActorRnnNetwork/batch_unflatten/strided_slice/stack�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2�
-ActorRnnNetwork/batch_unflatten/strided_sliceStridedSlice,ActorRnnNetwork/batch_flatten/Shape:output:0<ActorRnnNetwork/batch_unflatten/strided_slice/stack:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_1:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-ActorRnnNetwork/batch_unflatten/strided_slice�
%ActorRnnNetwork/batch_unflatten/ShapeShape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_unflatten/Shape�
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stack�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2�
/ActorRnnNetwork/batch_unflatten/strided_slice_1StridedSlice.ActorRnnNetwork/batch_unflatten/Shape:output:0>ActorRnnNetwork/batch_unflatten/strided_slice_1/stack:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask21
/ActorRnnNetwork/batch_unflatten/strided_slice_1�
+ActorRnnNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+ActorRnnNetwork/batch_unflatten/concat/axis�
&ActorRnnNetwork/batch_unflatten/concatConcatV26ActorRnnNetwork/batch_unflatten/strided_slice:output:08ActorRnnNetwork/batch_unflatten/strided_slice_1:output:04ActorRnnNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2(
&ActorRnnNetwork/batch_unflatten/concat�
'ActorRnnNetwork/batch_unflatten/ReshapeReshape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0/ActorRnnNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:����������2)
'ActorRnnNetwork/batch_unflatten/Reshape�
"ActorRnnNetwork/reset_mask/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ActorRnnNetwork/reset_mask/Equal/y�
 ActorRnnNetwork/reset_mask/EqualEqual%ActorRnnNetwork/ExpandDims_1:output:0+ActorRnnNetwork/reset_mask/Equal/y:output:0*
T0*'
_output_shapes
:���������2"
 ActorRnnNetwork/reset_mask/Equal�
#ActorRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#ActorRnnNetwork/dynamic_unroll/Rank�
*ActorRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/start�
*ActorRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/delta�
$ActorRnnNetwork/dynamic_unroll/rangeRange3ActorRnnNetwork/dynamic_unroll/range/start:output:0,ActorRnnNetwork/dynamic_unroll/Rank:output:03ActorRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/range�
.ActorRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       20
.ActorRnnNetwork/dynamic_unroll/concat/values_0�
*ActorRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ActorRnnNetwork/dynamic_unroll/concat/axis�
%ActorRnnNetwork/dynamic_unroll/concatConcatV27ActorRnnNetwork/dynamic_unroll/concat/values_0:output:0-ActorRnnNetwork/dynamic_unroll/range:output:03ActorRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%ActorRnnNetwork/dynamic_unroll/concat�
(ActorRnnNetwork/dynamic_unroll/transpose	Transpose0ActorRnnNetwork/batch_unflatten/Reshape:output:0.ActorRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:����������2*
(ActorRnnNetwork/dynamic_unroll/transpose�
$ActorRnnNetwork/dynamic_unroll/ShapeShape,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/Shape�
2ActorRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2ActorRnnNetwork/dynamic_unroll/strided_slice/stack�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2�
,ActorRnnNetwork/dynamic_unroll/strided_sliceStridedSlice-ActorRnnNetwork/dynamic_unroll/Shape:output:0;ActorRnnNetwork/dynamic_unroll/strided_slice/stack:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,ActorRnnNetwork/dynamic_unroll/strided_slice�
/ActorRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/ActorRnnNetwork/dynamic_unroll/transpose_1/perm�
*ActorRnnNetwork/dynamic_unroll/transpose_1	Transpose$ActorRnnNetwork/reset_mask/Equal:z:08ActorRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:���������2,
*ActorRnnNetwork/dynamic_unroll/transpose_1�
*ActorRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2,
*ActorRnnNetwork/dynamic_unroll/zeros/mul/y�
(ActorRnnNetwork/dynamic_unroll/zeros/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:03ActorRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(ActorRnnNetwork/dynamic_unroll/zeros/mul�
+ActorRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2-
+ActorRnnNetwork/dynamic_unroll/zeros/Less/y�
)ActorRnnNetwork/dynamic_unroll/zeros/LessLess,ActorRnnNetwork/dynamic_unroll/zeros/mul:z:04ActorRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)ActorRnnNetwork/dynamic_unroll/zeros/Less�
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2/
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1�
+ActorRnnNetwork/dynamic_unroll/zeros/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:06ActorRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+ActorRnnNetwork/dynamic_unroll/zeros/packed�
*ActorRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*ActorRnnNetwork/dynamic_unroll/zeros/Const�
$ActorRnnNetwork/dynamic_unroll/zerosFill4ActorRnnNetwork/dynamic_unroll/zeros/packed:output:03ActorRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:���������(2&
$ActorRnnNetwork/dynamic_unroll/zeros�
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y�
*ActorRnnNetwork/dynamic_unroll/zeros_1/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*ActorRnnNetwork/dynamic_unroll/zeros_1/mul�
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y�
+ActorRnnNetwork/dynamic_unroll/zeros_1/LessLess.ActorRnnNetwork/dynamic_unroll/zeros_1/mul:z:06ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+ActorRnnNetwork/dynamic_unroll/zeros_1/Less�
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(21
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1�
-ActorRnnNetwork/dynamic_unroll/zeros_1/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:08ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/packed�
,ActorRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/Const�
&ActorRnnNetwork/dynamic_unroll/zeros_1Fill6ActorRnnNetwork/dynamic_unroll/zeros_1/packed:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2(
&ActorRnnNetwork/dynamic_unroll/zeros_1�
&ActorRnnNetwork/dynamic_unroll/SqueezeSqueeze,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
 2(
&ActorRnnNetwork/dynamic_unroll/Squeeze�
(ActorRnnNetwork/dynamic_unroll/Squeeze_1Squeeze.ActorRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:���������*
squeeze_dims
 2*
(ActorRnnNetwork/dynamic_unroll/Squeeze_1�
%ActorRnnNetwork/dynamic_unroll/SelectSelect1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0-ActorRnnNetwork/dynamic_unroll/zeros:output:0SelectV2:output:0*
T0*'
_output_shapes
:���������(2'
%ActorRnnNetwork/dynamic_unroll/Select�
'ActorRnnNetwork/dynamic_unroll/Select_1Select1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0/ActorRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/dynamic_unroll/Select_1�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpGactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02@
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul/ActorRnnNetwork/dynamic_unroll/Squeeze:output:0FActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(�*
dtype02B
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul.ActorRnnNetwork/dynamic_unroll/Select:output:0HActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/addAddV29ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0;ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/add�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02A
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd0ActorRnnNetwork/dynamic_unroll/lstm_cell/add:z:0GActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd�
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/splitSplitAActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:09ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������(:���������(:���������(:���������(*
	num_split20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/split�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������(22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mulMul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:00ActorRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:���������(2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mul�
-ActorRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������(2/
-ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul4ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:01ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV20ActorRnnNetwork/dynamic_unroll/lstm_cell/mul:z:02ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������(21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:03ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2�
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dim�
)ActorRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:06ActorRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������(2+
)ActorRnnNetwork/dynamic_unroll/ExpandDims�
%ActorRnnNetwork/batch_flatten_1/ShapeShape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_flatten_1/Shape�
-ActorRnnNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����(   2/
-ActorRnnNetwork/batch_flatten_1/Reshape/shape�
'ActorRnnNetwork/batch_flatten_1/ReshapeReshape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:06ActorRnnNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/batch_flatten_1/Reshape�
ActorRnnNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����(   2!
ActorRnnNetwork/flatten_1/Const�
!ActorRnnNetwork/flatten_1/ReshapeReshape0ActorRnnNetwork/batch_flatten_1/Reshape:output:0(ActorRnnNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������(2#
!ActorRnnNetwork/flatten_1/Reshape�
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpReadVariableOp;actorrnnnetwork_output_dense_matmul_readvariableop_resource*
_output_shapes

:(2*
dtype024
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp�
#ActorRnnNetwork/output/dense/MatMulMatMul*ActorRnnNetwork/flatten_1/Reshape:output:0:ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22%
#ActorRnnNetwork/output/dense/MatMul�
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOpReadVariableOp<actorrnnnetwork_output_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype025
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�
$ActorRnnNetwork/output/dense/BiasAddBiasAdd-ActorRnnNetwork/output/dense/MatMul:product:0;ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22&
$ActorRnnNetwork/output/dense/BiasAdd�
!ActorRnnNetwork/output/dense/ReluRelu-ActorRnnNetwork/output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������22#
!ActorRnnNetwork/output/dense/Relu�
,ActorRnnNetwork/action/MatMul/ReadVariableOpReadVariableOp5actorrnnnetwork_action_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,ActorRnnNetwork/action/MatMul/ReadVariableOp�
ActorRnnNetwork/action/MatMulMatMul/ActorRnnNetwork/output/dense/Relu:activations:04ActorRnnNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/MatMul�
-ActorRnnNetwork/action/BiasAdd/ReadVariableOpReadVariableOp6actorrnnnetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�
ActorRnnNetwork/action/BiasAddBiasAdd'ActorRnnNetwork/action/MatMul:product:05ActorRnnNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
ActorRnnNetwork/action/BiasAdd�
ActorRnnNetwork/action/TanhTanh'ActorRnnNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/Tanh�
ActorRnnNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/Reshape/shape�
ActorRnnNetwork/ReshapeReshapeActorRnnNetwork/action/Tanh:y:0&ActorRnnNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/Reshapes
ActorRnnNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/mul/x�
ActorRnnNetwork/mulMulActorRnnNetwork/mul/x:output:0 ActorRnnNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/muls
ActorRnnNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/add/x�
ActorRnnNetwork/addAddV2ActorRnnNetwork/add/x:output:0ActorRnnNetwork/mul:z:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/add�
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stack�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2�
/ActorRnnNetwork/batch_unflatten_1/strided_sliceStridedSlice.ActorRnnNetwork/batch_flatten_1/Shape:output:0>ActorRnnNetwork/batch_unflatten_1/strided_slice/stack:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask21
/ActorRnnNetwork/batch_unflatten_1/strided_slice�
'ActorRnnNetwork/batch_unflatten_1/ShapeShapeActorRnnNetwork/add:z:0*
T0*
_output_shapes
:*
out_type0	2)
'ActorRnnNetwork/batch_unflatten_1/Shape�
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2�
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1StridedSlice0ActorRnnNetwork/batch_unflatten_1/Shape:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask23
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1�
-ActorRnnNetwork/batch_unflatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-ActorRnnNetwork/batch_unflatten_1/concat/axis�
(ActorRnnNetwork/batch_unflatten_1/concatConcatV28ActorRnnNetwork/batch_unflatten_1/strided_slice:output:0:ActorRnnNetwork/batch_unflatten_1/strided_slice_1:output:06ActorRnnNetwork/batch_unflatten_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:2*
(ActorRnnNetwork/batch_unflatten_1/concat�
)ActorRnnNetwork/batch_unflatten_1/ReshapeReshapeActorRnnNetwork/add:z:01ActorRnnNetwork/batch_unflatten_1/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:���������2+
)ActorRnnNetwork/batch_unflatten_1/Reshape�
ActorRnnNetwork/SqueezeSqueeze2ActorRnnNetwork/batch_unflatten_1/Reshape:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
2
ActorRnnNetwork/Squeezem
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtole
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identity ActorRnnNetwork/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1i

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_3�

Identity_4Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_4�
NoOpNoOp.^ActorRnnNetwork/action/BiasAdd/ReadVariableOp-^ActorRnnNetwork/action/MatMul/ReadVariableOp@^ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpA^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp7^ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp9^ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp8^ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4^ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3^ActorRnnNetwork/output/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 2^
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp-ActorRnnNetwork/action/BiasAdd/ReadVariableOp2\
,ActorRnnNetwork/action/MatMul/ReadVariableOp,ActorRnnNetwork/action/MatMul/ReadVariableOp2�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2p
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2t
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp2r
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2j
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp2h
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation:JF
'
_output_shapes
:���������(

_user_specified_name0:JF
'
_output_shapes
:���������(

_user_specified_name1
�
[
+__inference_function_with_signature_3405106

batch_size
identity

identity_1�
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_get_initial_state_34051012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������(2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������(2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
'
%__inference_signature_wrapper_6872160�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_34051332
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
��
�
__inference_action_3405015
	time_step
time_step_1
time_step_2
time_step_3
policy_state
policy_state_1Q
>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource:	�N
?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource:	�T
@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��P
Aactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�[
Gactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
��\
Iactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(�W
Hactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	�M
;actorrnnnetwork_output_dense_matmul_readvariableop_resource:(2J
<actorrnnnetwork_output_dense_biasadd_readvariableop_resource:2G
5actorrnnnetwork_action_matmul_readvariableop_resource:2D
6actorrnnnetwork_action_biasadd_readvariableop_resource:
identity

identity_1

identity_2��-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�,ActorRnnNetwork/action/MatMul/ReadVariableOp�?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpI
ShapeShapetime_step_2*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:���������2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis�
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:���������2	
Reshape�
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*'
_output_shapes
:���������(2

SelectV2�

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*'
_output_shapes
:���������(2

SelectV2_1�
ActorRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
ActorRnnNetwork/ExpandDims/dim�
ActorRnnNetwork/ExpandDims
ExpandDimstime_step_3'ActorRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims�
 ActorRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 ActorRnnNetwork/ExpandDims_1/dim�
ActorRnnNetwork/ExpandDims_1
ExpandDims	time_step)ActorRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims_1�
#ActorRnnNetwork/batch_flatten/ShapeShape#ActorRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2%
#ActorRnnNetwork/batch_flatten/Shape�
+ActorRnnNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2-
+ActorRnnNetwork/batch_flatten/Reshape/shape�
%ActorRnnNetwork/batch_flatten/ReshapeReshape#ActorRnnNetwork/ExpandDims:output:04ActorRnnNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2'
%ActorRnnNetwork/batch_flatten/Reshape�
ActorRnnNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/flatten/Const�
ActorRnnNetwork/flatten/ReshapeReshape.ActorRnnNetwork/batch_flatten/Reshape:output:0&ActorRnnNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2!
ActorRnnNetwork/flatten/Reshape�
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype027
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�
&ActorRnnNetwork/input_mlp/dense/MatMulMatMul(ActorRnnNetwork/flatten/Reshape:output:0=ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/MatMul�
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�
'ActorRnnNetwork/input_mlp/dense/BiasAddBiasAdd0ActorRnnNetwork/input_mlp/dense/MatMul:product:0>ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'ActorRnnNetwork/input_mlp/dense/BiasAdd�
$ActorRnnNetwork/input_mlp/dense/ReluRelu0ActorRnnNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2&
$ActorRnnNetwork/input_mlp/dense/Relu�
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�
(ActorRnnNetwork/input_mlp/dense/MatMul_1MatMul2ActorRnnNetwork/input_mlp/dense/Relu:activations:0?ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(ActorRnnNetwork/input_mlp/dense/MatMul_1�
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOpAactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1BiasAdd2ActorRnnNetwork/input_mlp/dense/MatMul_1:product:0@ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1�
&ActorRnnNetwork/input_mlp/dense/Relu_1Relu2ActorRnnNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/Relu_1�
3ActorRnnNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3ActorRnnNetwork/batch_unflatten/strided_slice/stack�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2�
-ActorRnnNetwork/batch_unflatten/strided_sliceStridedSlice,ActorRnnNetwork/batch_flatten/Shape:output:0<ActorRnnNetwork/batch_unflatten/strided_slice/stack:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_1:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-ActorRnnNetwork/batch_unflatten/strided_slice�
%ActorRnnNetwork/batch_unflatten/ShapeShape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_unflatten/Shape�
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stack�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2�
/ActorRnnNetwork/batch_unflatten/strided_slice_1StridedSlice.ActorRnnNetwork/batch_unflatten/Shape:output:0>ActorRnnNetwork/batch_unflatten/strided_slice_1/stack:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask21
/ActorRnnNetwork/batch_unflatten/strided_slice_1�
+ActorRnnNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+ActorRnnNetwork/batch_unflatten/concat/axis�
&ActorRnnNetwork/batch_unflatten/concatConcatV26ActorRnnNetwork/batch_unflatten/strided_slice:output:08ActorRnnNetwork/batch_unflatten/strided_slice_1:output:04ActorRnnNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2(
&ActorRnnNetwork/batch_unflatten/concat�
'ActorRnnNetwork/batch_unflatten/ReshapeReshape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0/ActorRnnNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:����������2)
'ActorRnnNetwork/batch_unflatten/Reshape�
"ActorRnnNetwork/reset_mask/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ActorRnnNetwork/reset_mask/Equal/y�
 ActorRnnNetwork/reset_mask/EqualEqual%ActorRnnNetwork/ExpandDims_1:output:0+ActorRnnNetwork/reset_mask/Equal/y:output:0*
T0*'
_output_shapes
:���������2"
 ActorRnnNetwork/reset_mask/Equal�
#ActorRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#ActorRnnNetwork/dynamic_unroll/Rank�
*ActorRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/start�
*ActorRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/delta�
$ActorRnnNetwork/dynamic_unroll/rangeRange3ActorRnnNetwork/dynamic_unroll/range/start:output:0,ActorRnnNetwork/dynamic_unroll/Rank:output:03ActorRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/range�
.ActorRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       20
.ActorRnnNetwork/dynamic_unroll/concat/values_0�
*ActorRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ActorRnnNetwork/dynamic_unroll/concat/axis�
%ActorRnnNetwork/dynamic_unroll/concatConcatV27ActorRnnNetwork/dynamic_unroll/concat/values_0:output:0-ActorRnnNetwork/dynamic_unroll/range:output:03ActorRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%ActorRnnNetwork/dynamic_unroll/concat�
(ActorRnnNetwork/dynamic_unroll/transpose	Transpose0ActorRnnNetwork/batch_unflatten/Reshape:output:0.ActorRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:����������2*
(ActorRnnNetwork/dynamic_unroll/transpose�
$ActorRnnNetwork/dynamic_unroll/ShapeShape,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/Shape�
2ActorRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2ActorRnnNetwork/dynamic_unroll/strided_slice/stack�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2�
,ActorRnnNetwork/dynamic_unroll/strided_sliceStridedSlice-ActorRnnNetwork/dynamic_unroll/Shape:output:0;ActorRnnNetwork/dynamic_unroll/strided_slice/stack:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,ActorRnnNetwork/dynamic_unroll/strided_slice�
/ActorRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/ActorRnnNetwork/dynamic_unroll/transpose_1/perm�
*ActorRnnNetwork/dynamic_unroll/transpose_1	Transpose$ActorRnnNetwork/reset_mask/Equal:z:08ActorRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:���������2,
*ActorRnnNetwork/dynamic_unroll/transpose_1�
*ActorRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2,
*ActorRnnNetwork/dynamic_unroll/zeros/mul/y�
(ActorRnnNetwork/dynamic_unroll/zeros/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:03ActorRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(ActorRnnNetwork/dynamic_unroll/zeros/mul�
+ActorRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2-
+ActorRnnNetwork/dynamic_unroll/zeros/Less/y�
)ActorRnnNetwork/dynamic_unroll/zeros/LessLess,ActorRnnNetwork/dynamic_unroll/zeros/mul:z:04ActorRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)ActorRnnNetwork/dynamic_unroll/zeros/Less�
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2/
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1�
+ActorRnnNetwork/dynamic_unroll/zeros/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:06ActorRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+ActorRnnNetwork/dynamic_unroll/zeros/packed�
*ActorRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*ActorRnnNetwork/dynamic_unroll/zeros/Const�
$ActorRnnNetwork/dynamic_unroll/zerosFill4ActorRnnNetwork/dynamic_unroll/zeros/packed:output:03ActorRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:���������(2&
$ActorRnnNetwork/dynamic_unroll/zeros�
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y�
*ActorRnnNetwork/dynamic_unroll/zeros_1/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*ActorRnnNetwork/dynamic_unroll/zeros_1/mul�
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y�
+ActorRnnNetwork/dynamic_unroll/zeros_1/LessLess.ActorRnnNetwork/dynamic_unroll/zeros_1/mul:z:06ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+ActorRnnNetwork/dynamic_unroll/zeros_1/Less�
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(21
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1�
-ActorRnnNetwork/dynamic_unroll/zeros_1/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:08ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/packed�
,ActorRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/Const�
&ActorRnnNetwork/dynamic_unroll/zeros_1Fill6ActorRnnNetwork/dynamic_unroll/zeros_1/packed:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2(
&ActorRnnNetwork/dynamic_unroll/zeros_1�
&ActorRnnNetwork/dynamic_unroll/SqueezeSqueeze,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
 2(
&ActorRnnNetwork/dynamic_unroll/Squeeze�
(ActorRnnNetwork/dynamic_unroll/Squeeze_1Squeeze.ActorRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:���������*
squeeze_dims
 2*
(ActorRnnNetwork/dynamic_unroll/Squeeze_1�
%ActorRnnNetwork/dynamic_unroll/SelectSelect1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0-ActorRnnNetwork/dynamic_unroll/zeros:output:0SelectV2:output:0*
T0*'
_output_shapes
:���������(2'
%ActorRnnNetwork/dynamic_unroll/Select�
'ActorRnnNetwork/dynamic_unroll/Select_1Select1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0/ActorRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/dynamic_unroll/Select_1�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpGactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02@
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul/ActorRnnNetwork/dynamic_unroll/Squeeze:output:0FActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(�*
dtype02B
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul.ActorRnnNetwork/dynamic_unroll/Select:output:0HActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/addAddV29ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0;ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/add�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02A
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd0ActorRnnNetwork/dynamic_unroll/lstm_cell/add:z:0GActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd�
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/splitSplitAActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:09ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������(:���������(:���������(:���������(*
	num_split20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/split�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������(22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mulMul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:00ActorRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:���������(2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mul�
-ActorRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������(2/
-ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul4ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:01ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV20ActorRnnNetwork/dynamic_unroll/lstm_cell/mul:z:02ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������(21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:03ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2�
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dim�
)ActorRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:06ActorRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������(2+
)ActorRnnNetwork/dynamic_unroll/ExpandDims�
%ActorRnnNetwork/batch_flatten_1/ShapeShape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_flatten_1/Shape�
-ActorRnnNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����(   2/
-ActorRnnNetwork/batch_flatten_1/Reshape/shape�
'ActorRnnNetwork/batch_flatten_1/ReshapeReshape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:06ActorRnnNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/batch_flatten_1/Reshape�
ActorRnnNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����(   2!
ActorRnnNetwork/flatten_1/Const�
!ActorRnnNetwork/flatten_1/ReshapeReshape0ActorRnnNetwork/batch_flatten_1/Reshape:output:0(ActorRnnNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������(2#
!ActorRnnNetwork/flatten_1/Reshape�
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpReadVariableOp;actorrnnnetwork_output_dense_matmul_readvariableop_resource*
_output_shapes

:(2*
dtype024
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp�
#ActorRnnNetwork/output/dense/MatMulMatMul*ActorRnnNetwork/flatten_1/Reshape:output:0:ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22%
#ActorRnnNetwork/output/dense/MatMul�
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOpReadVariableOp<actorrnnnetwork_output_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype025
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�
$ActorRnnNetwork/output/dense/BiasAddBiasAdd-ActorRnnNetwork/output/dense/MatMul:product:0;ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22&
$ActorRnnNetwork/output/dense/BiasAdd�
!ActorRnnNetwork/output/dense/ReluRelu-ActorRnnNetwork/output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������22#
!ActorRnnNetwork/output/dense/Relu�
,ActorRnnNetwork/action/MatMul/ReadVariableOpReadVariableOp5actorrnnnetwork_action_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,ActorRnnNetwork/action/MatMul/ReadVariableOp�
ActorRnnNetwork/action/MatMulMatMul/ActorRnnNetwork/output/dense/Relu:activations:04ActorRnnNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/MatMul�
-ActorRnnNetwork/action/BiasAdd/ReadVariableOpReadVariableOp6actorrnnnetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�
ActorRnnNetwork/action/BiasAddBiasAdd'ActorRnnNetwork/action/MatMul:product:05ActorRnnNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
ActorRnnNetwork/action/BiasAdd�
ActorRnnNetwork/action/TanhTanh'ActorRnnNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/Tanh�
ActorRnnNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/Reshape/shape�
ActorRnnNetwork/ReshapeReshapeActorRnnNetwork/action/Tanh:y:0&ActorRnnNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/Reshapes
ActorRnnNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/mul/x�
ActorRnnNetwork/mulMulActorRnnNetwork/mul/x:output:0 ActorRnnNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/muls
ActorRnnNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/add/x�
ActorRnnNetwork/addAddV2ActorRnnNetwork/add/x:output:0ActorRnnNetwork/mul:z:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/add�
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stack�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2�
/ActorRnnNetwork/batch_unflatten_1/strided_sliceStridedSlice.ActorRnnNetwork/batch_flatten_1/Shape:output:0>ActorRnnNetwork/batch_unflatten_1/strided_slice/stack:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask21
/ActorRnnNetwork/batch_unflatten_1/strided_slice�
'ActorRnnNetwork/batch_unflatten_1/ShapeShapeActorRnnNetwork/add:z:0*
T0*
_output_shapes
:*
out_type0	2)
'ActorRnnNetwork/batch_unflatten_1/Shape�
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2�
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1StridedSlice0ActorRnnNetwork/batch_unflatten_1/Shape:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask23
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1�
-ActorRnnNetwork/batch_unflatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-ActorRnnNetwork/batch_unflatten_1/concat/axis�
(ActorRnnNetwork/batch_unflatten_1/concatConcatV28ActorRnnNetwork/batch_unflatten_1/strided_slice:output:0:ActorRnnNetwork/batch_unflatten_1/strided_slice_1:output:06ActorRnnNetwork/batch_unflatten_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:2*
(ActorRnnNetwork/batch_unflatten_1/concat�
)ActorRnnNetwork/batch_unflatten_1/ReshapeReshapeActorRnnNetwork/add:z:01ActorRnnNetwork/batch_unflatten_1/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:���������2+
)ActorRnnNetwork/batch_unflatten_1/Reshape�
ActorRnnNetwork/SqueezeSqueeze2ActorRnnNetwork/batch_unflatten_1/Reshape:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
2
ActorRnnNetwork/Squeezem
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShape ActorRnnNetwork/Squeeze:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastTo ActorRnnNetwork/Squeeze:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:���������2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_1�

Identity_2Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_2�
NoOpNoOp.^ActorRnnNetwork/action/BiasAdd/ReadVariableOp-^ActorRnnNetwork/action/MatMul/ReadVariableOp@^ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpA^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp7^ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp9^ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp8^ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4^ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3^ActorRnnNetwork/output/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 2^
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp-ActorRnnNetwork/action/BiasAdd/ReadVariableOp2\
,ActorRnnNetwork/action/MatMul/ReadVariableOp,ActorRnnNetwork/action/MatMul/ReadVariableOp2�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2p
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2t
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp2r
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2j
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp2h
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step:UQ
'
_output_shapes
:���������(
&
_user_specified_namepolicy_state:UQ
'
_output_shapes
:���������(
&
_user_specified_namepolicy_state
��
�
__inference_action_3405552
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
policy_state_0
policy_state_1Q
>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource:	�N
?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource:	�T
@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��P
Aactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�[
Gactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
��\
Iactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(�W
Hactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	�M
;actorrnnnetwork_output_dense_matmul_readvariableop_resource:(2J
<actorrnnnetwork_output_dense_biasadd_readvariableop_resource:2G
5actorrnnnetwork_action_matmul_readvariableop_resource:2D
6actorrnnnetwork_action_biasadd_readvariableop_resource:
identity

identity_1

identity_2��-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�,ActorRnnNetwork/action/MatMul/ReadVariableOp�?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpP
ShapeShapetime_step_discount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yl
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:���������2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis�
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:���������2	
Reshape�
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state_0*
T0*'
_output_shapes
:���������(2

SelectV2�

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*'
_output_shapes
:���������(2

SelectV2_1�
ActorRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
ActorRnnNetwork/ExpandDims/dim�
ActorRnnNetwork/ExpandDims
ExpandDimstime_step_observation'ActorRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims�
 ActorRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 ActorRnnNetwork/ExpandDims_1/dim�
ActorRnnNetwork/ExpandDims_1
ExpandDimstime_step_step_type)ActorRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/ExpandDims_1�
#ActorRnnNetwork/batch_flatten/ShapeShape#ActorRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2%
#ActorRnnNetwork/batch_flatten/Shape�
+ActorRnnNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2-
+ActorRnnNetwork/batch_flatten/Reshape/shape�
%ActorRnnNetwork/batch_flatten/ReshapeReshape#ActorRnnNetwork/ExpandDims:output:04ActorRnnNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2'
%ActorRnnNetwork/batch_flatten/Reshape�
ActorRnnNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/flatten/Const�
ActorRnnNetwork/flatten/ReshapeReshape.ActorRnnNetwork/batch_flatten/Reshape:output:0&ActorRnnNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2!
ActorRnnNetwork/flatten/Reshape�
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp>actorrnnnetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype027
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp�
&ActorRnnNetwork/input_mlp/dense/MatMulMatMul(ActorRnnNetwork/flatten/Reshape:output:0=ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/MatMul�
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp?actorrnnnetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�
'ActorRnnNetwork/input_mlp/dense/BiasAddBiasAdd0ActorRnnNetwork/input_mlp/dense/MatMul:product:0>ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'ActorRnnNetwork/input_mlp/dense/BiasAdd�
$ActorRnnNetwork/input_mlp/dense/ReluRelu0ActorRnnNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2&
$ActorRnnNetwork/input_mlp/dense/Relu�
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp@actorrnnnetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�
(ActorRnnNetwork/input_mlp/dense/MatMul_1MatMul2ActorRnnNetwork/input_mlp/dense/Relu:activations:0?ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(ActorRnnNetwork/input_mlp/dense/MatMul_1�
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOpAactorrnnnetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1BiasAdd2ActorRnnNetwork/input_mlp/dense/MatMul_1:product:0@ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)ActorRnnNetwork/input_mlp/dense/BiasAdd_1�
&ActorRnnNetwork/input_mlp/dense/Relu_1Relu2ActorRnnNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2(
&ActorRnnNetwork/input_mlp/dense/Relu_1�
3ActorRnnNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3ActorRnnNetwork/batch_unflatten/strided_slice/stack�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_1�
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice/stack_2�
-ActorRnnNetwork/batch_unflatten/strided_sliceStridedSlice,ActorRnnNetwork/batch_flatten/Shape:output:0<ActorRnnNetwork/batch_unflatten/strided_slice/stack:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_1:output:0>ActorRnnNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-ActorRnnNetwork/batch_unflatten/strided_slice�
%ActorRnnNetwork/batch_unflatten/ShapeShape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_unflatten/Shape�
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5ActorRnnNetwork/batch_unflatten/strided_slice_1/stack�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1�
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2�
/ActorRnnNetwork/batch_unflatten/strided_slice_1StridedSlice.ActorRnnNetwork/batch_unflatten/Shape:output:0>ActorRnnNetwork/batch_unflatten/strided_slice_1/stack:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_1:output:0@ActorRnnNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask21
/ActorRnnNetwork/batch_unflatten/strided_slice_1�
+ActorRnnNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+ActorRnnNetwork/batch_unflatten/concat/axis�
&ActorRnnNetwork/batch_unflatten/concatConcatV26ActorRnnNetwork/batch_unflatten/strided_slice:output:08ActorRnnNetwork/batch_unflatten/strided_slice_1:output:04ActorRnnNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2(
&ActorRnnNetwork/batch_unflatten/concat�
'ActorRnnNetwork/batch_unflatten/ReshapeReshape4ActorRnnNetwork/input_mlp/dense/Relu_1:activations:0/ActorRnnNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:����������2)
'ActorRnnNetwork/batch_unflatten/Reshape�
"ActorRnnNetwork/reset_mask/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ActorRnnNetwork/reset_mask/Equal/y�
 ActorRnnNetwork/reset_mask/EqualEqual%ActorRnnNetwork/ExpandDims_1:output:0+ActorRnnNetwork/reset_mask/Equal/y:output:0*
T0*'
_output_shapes
:���������2"
 ActorRnnNetwork/reset_mask/Equal�
#ActorRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#ActorRnnNetwork/dynamic_unroll/Rank�
*ActorRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/start�
*ActorRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*ActorRnnNetwork/dynamic_unroll/range/delta�
$ActorRnnNetwork/dynamic_unroll/rangeRange3ActorRnnNetwork/dynamic_unroll/range/start:output:0,ActorRnnNetwork/dynamic_unroll/Rank:output:03ActorRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/range�
.ActorRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       20
.ActorRnnNetwork/dynamic_unroll/concat/values_0�
*ActorRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ActorRnnNetwork/dynamic_unroll/concat/axis�
%ActorRnnNetwork/dynamic_unroll/concatConcatV27ActorRnnNetwork/dynamic_unroll/concat/values_0:output:0-ActorRnnNetwork/dynamic_unroll/range:output:03ActorRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%ActorRnnNetwork/dynamic_unroll/concat�
(ActorRnnNetwork/dynamic_unroll/transpose	Transpose0ActorRnnNetwork/batch_unflatten/Reshape:output:0.ActorRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:����������2*
(ActorRnnNetwork/dynamic_unroll/transpose�
$ActorRnnNetwork/dynamic_unroll/ShapeShape,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2&
$ActorRnnNetwork/dynamic_unroll/Shape�
2ActorRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2ActorRnnNetwork/dynamic_unroll/strided_slice/stack�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1�
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2�
,ActorRnnNetwork/dynamic_unroll/strided_sliceStridedSlice-ActorRnnNetwork/dynamic_unroll/Shape:output:0;ActorRnnNetwork/dynamic_unroll/strided_slice/stack:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0=ActorRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,ActorRnnNetwork/dynamic_unroll/strided_slice�
/ActorRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/ActorRnnNetwork/dynamic_unroll/transpose_1/perm�
*ActorRnnNetwork/dynamic_unroll/transpose_1	Transpose$ActorRnnNetwork/reset_mask/Equal:z:08ActorRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:���������2,
*ActorRnnNetwork/dynamic_unroll/transpose_1�
*ActorRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2,
*ActorRnnNetwork/dynamic_unroll/zeros/mul/y�
(ActorRnnNetwork/dynamic_unroll/zeros/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:03ActorRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(ActorRnnNetwork/dynamic_unroll/zeros/mul�
+ActorRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2-
+ActorRnnNetwork/dynamic_unroll/zeros/Less/y�
)ActorRnnNetwork/dynamic_unroll/zeros/LessLess,ActorRnnNetwork/dynamic_unroll/zeros/mul:z:04ActorRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)ActorRnnNetwork/dynamic_unroll/zeros/Less�
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2/
-ActorRnnNetwork/dynamic_unroll/zeros/packed/1�
+ActorRnnNetwork/dynamic_unroll/zeros/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:06ActorRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+ActorRnnNetwork/dynamic_unroll/zeros/packed�
*ActorRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*ActorRnnNetwork/dynamic_unroll/zeros/Const�
$ActorRnnNetwork/dynamic_unroll/zerosFill4ActorRnnNetwork/dynamic_unroll/zeros/packed:output:03ActorRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:���������(2&
$ActorRnnNetwork/dynamic_unroll/zeros�
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y�
*ActorRnnNetwork/dynamic_unroll/zeros_1/mulMul5ActorRnnNetwork/dynamic_unroll/strided_slice:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*ActorRnnNetwork/dynamic_unroll/zeros_1/mul�
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y�
+ActorRnnNetwork/dynamic_unroll/zeros_1/LessLess.ActorRnnNetwork/dynamic_unroll/zeros_1/mul:z:06ActorRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+ActorRnnNetwork/dynamic_unroll/zeros_1/Less�
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(21
/ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1�
-ActorRnnNetwork/dynamic_unroll/zeros_1/packedPack5ActorRnnNetwork/dynamic_unroll/strided_slice:output:08ActorRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-ActorRnnNetwork/dynamic_unroll/zeros_1/packed�
,ActorRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,ActorRnnNetwork/dynamic_unroll/zeros_1/Const�
&ActorRnnNetwork/dynamic_unroll/zeros_1Fill6ActorRnnNetwork/dynamic_unroll/zeros_1/packed:output:05ActorRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2(
&ActorRnnNetwork/dynamic_unroll/zeros_1�
&ActorRnnNetwork/dynamic_unroll/SqueezeSqueeze,ActorRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
 2(
&ActorRnnNetwork/dynamic_unroll/Squeeze�
(ActorRnnNetwork/dynamic_unroll/Squeeze_1Squeeze.ActorRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:���������*
squeeze_dims
 2*
(ActorRnnNetwork/dynamic_unroll/Squeeze_1�
%ActorRnnNetwork/dynamic_unroll/SelectSelect1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0-ActorRnnNetwork/dynamic_unroll/zeros:output:0SelectV2:output:0*
T0*'
_output_shapes
:���������(2'
%ActorRnnNetwork/dynamic_unroll/Select�
'ActorRnnNetwork/dynamic_unroll/Select_1Select1ActorRnnNetwork/dynamic_unroll/Squeeze_1:output:0/ActorRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/dynamic_unroll/Select_1�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpGactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02@
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul/ActorRnnNetwork/dynamic_unroll/Squeeze:output:0FActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIactorrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(�*
dtype02B
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp�
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul.ActorRnnNetwork/dynamic_unroll/Select:output:0HActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/addAddV29ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0;ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/add�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHactorrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02A
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd0ActorRnnNetwork/dynamic_unroll/lstm_cell/add:z:0GActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd�
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8ActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/splitSplitAActorRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:09ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������(:���������(:���������(:���������(*
	num_split20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/split�
0ActorRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������(22
0ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1�
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mulMul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:00ActorRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:���������(2.
,ActorRnnNetwork/dynamic_unroll/lstm_cell/mul�
-ActorRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������(2/
-ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul4ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:01ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV20ActorRnnNetwork/dynamic_unroll/lstm_cell/mul:z:02ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1�
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid7ActorRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������(24
2ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2�
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������(21
/ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1�
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul6ActorRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:03ActorRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:���������(20
.ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2�
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-ActorRnnNetwork/dynamic_unroll/ExpandDims/dim�
)ActorRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:06ActorRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������(2+
)ActorRnnNetwork/dynamic_unroll/ExpandDims�
%ActorRnnNetwork/batch_flatten_1/ShapeShape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2'
%ActorRnnNetwork/batch_flatten_1/Shape�
-ActorRnnNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����(   2/
-ActorRnnNetwork/batch_flatten_1/Reshape/shape�
'ActorRnnNetwork/batch_flatten_1/ReshapeReshape2ActorRnnNetwork/dynamic_unroll/ExpandDims:output:06ActorRnnNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������(2)
'ActorRnnNetwork/batch_flatten_1/Reshape�
ActorRnnNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����(   2!
ActorRnnNetwork/flatten_1/Const�
!ActorRnnNetwork/flatten_1/ReshapeReshape0ActorRnnNetwork/batch_flatten_1/Reshape:output:0(ActorRnnNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������(2#
!ActorRnnNetwork/flatten_1/Reshape�
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOpReadVariableOp;actorrnnnetwork_output_dense_matmul_readvariableop_resource*
_output_shapes

:(2*
dtype024
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp�
#ActorRnnNetwork/output/dense/MatMulMatMul*ActorRnnNetwork/flatten_1/Reshape:output:0:ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22%
#ActorRnnNetwork/output/dense/MatMul�
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOpReadVariableOp<actorrnnnetwork_output_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype025
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp�
$ActorRnnNetwork/output/dense/BiasAddBiasAdd-ActorRnnNetwork/output/dense/MatMul:product:0;ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22&
$ActorRnnNetwork/output/dense/BiasAdd�
!ActorRnnNetwork/output/dense/ReluRelu-ActorRnnNetwork/output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������22#
!ActorRnnNetwork/output/dense/Relu�
,ActorRnnNetwork/action/MatMul/ReadVariableOpReadVariableOp5actorrnnnetwork_action_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,ActorRnnNetwork/action/MatMul/ReadVariableOp�
ActorRnnNetwork/action/MatMulMatMul/ActorRnnNetwork/output/dense/Relu:activations:04ActorRnnNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/MatMul�
-ActorRnnNetwork/action/BiasAdd/ReadVariableOpReadVariableOp6actorrnnnetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp�
ActorRnnNetwork/action/BiasAddBiasAdd'ActorRnnNetwork/action/MatMul:product:05ActorRnnNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
ActorRnnNetwork/action/BiasAdd�
ActorRnnNetwork/action/TanhTanh'ActorRnnNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/action/Tanh�
ActorRnnNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
ActorRnnNetwork/Reshape/shape�
ActorRnnNetwork/ReshapeReshapeActorRnnNetwork/action/Tanh:y:0&ActorRnnNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/Reshapes
ActorRnnNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/mul/x�
ActorRnnNetwork/mulMulActorRnnNetwork/mul/x:output:0 ActorRnnNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/muls
ActorRnnNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ActorRnnNetwork/add/x�
ActorRnnNetwork/addAddV2ActorRnnNetwork/add/x:output:0ActorRnnNetwork/mul:z:0*
T0*'
_output_shapes
:���������2
ActorRnnNetwork/add�
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5ActorRnnNetwork/batch_unflatten_1/strided_slice/stack�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1�
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2�
/ActorRnnNetwork/batch_unflatten_1/strided_sliceStridedSlice.ActorRnnNetwork/batch_flatten_1/Shape:output:0>ActorRnnNetwork/batch_unflatten_1/strided_slice/stack:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_1:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask21
/ActorRnnNetwork/batch_unflatten_1/strided_slice�
'ActorRnnNetwork/batch_unflatten_1/ShapeShapeActorRnnNetwork/add:z:0*
T0*
_output_shapes
:*
out_type0	2)
'ActorRnnNetwork/batch_unflatten_1/Shape�
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1�
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2�
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1StridedSlice0ActorRnnNetwork/batch_unflatten_1/Shape:output:0@ActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_1:output:0BActorRnnNetwork/batch_unflatten_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask23
1ActorRnnNetwork/batch_unflatten_1/strided_slice_1�
-ActorRnnNetwork/batch_unflatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-ActorRnnNetwork/batch_unflatten_1/concat/axis�
(ActorRnnNetwork/batch_unflatten_1/concatConcatV28ActorRnnNetwork/batch_unflatten_1/strided_slice:output:0:ActorRnnNetwork/batch_unflatten_1/strided_slice_1:output:06ActorRnnNetwork/batch_unflatten_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:2*
(ActorRnnNetwork/batch_unflatten_1/concat�
)ActorRnnNetwork/batch_unflatten_1/ReshapeReshapeActorRnnNetwork/add:z:01ActorRnnNetwork/batch_unflatten_1/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:���������2+
)ActorRnnNetwork/batch_unflatten_1/Reshape�
ActorRnnNetwork/SqueezeSqueeze2ActorRnnNetwork/batch_unflatten_1/Reshape:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
2
ActorRnnNetwork/Squeezem
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShape ActorRnnNetwork/Squeeze:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice�
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0�
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastTo ActorRnnNetwork/Squeeze:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack�
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1�
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:���������2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_1�

Identity_2Identity2ActorRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity_2�
NoOpNoOp.^ActorRnnNetwork/action/BiasAdd/ReadVariableOp-^ActorRnnNetwork/action/MatMul/ReadVariableOp@^ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpA^ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp7^ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp9^ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp8^ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4^ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3^ActorRnnNetwork/output/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������:���������:���������:���������:���������(:���������(: : : : : : : : : : : 2^
-ActorRnnNetwork/action/BiasAdd/ReadVariableOp-ActorRnnNetwork/action/BiasAdd/ReadVariableOp2\
,ActorRnnNetwork/action/MatMul/ReadVariableOp,ActorRnnNetwork/action/MatMul/ReadVariableOp2�
?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?ActorRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2�
>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp>ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2�
@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp@ActorRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2p
6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6ActorRnnNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2t
8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp8ActorRnnNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp5ActorRnnNetwork/input_mlp/dense/MatMul/ReadVariableOp2r
7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp7ActorRnnNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2j
3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp3ActorRnnNetwork/output/dense/BiasAdd/ReadVariableOp2h
2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp2ActorRnnNetwork/output/dense/MatMul/ReadVariableOp:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step/observation:WS
'
_output_shapes
:���������(
(
_user_specified_namepolicy_state/0:WS
'
_output_shapes
:���������(
(
_user_specified_namepolicy_state/1
�
b
__inference_<lambda>_161618!
readvariableop_resource:	 
identity	��ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOp`
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�(
�
 __inference__traced_save_6872230
file_prefix'
#savev2_variable_read_readvariableop	<
8savev2_actorrnnnetwork_action_kernel_read_readvariableop:
6savev2_actorrnnnetwork_action_bias_read_readvariableopD
@savev2_actorrnnnetwork_dynamic_unroll_kernel_read_readvariableopN
Jsavev2_actorrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableopB
>savev2_actorrnnnetwork_dynamic_unroll_bias_read_readvariableopE
Asavev2_actorrnnnetwork_input_mlp_dense_kernel_read_readvariableopC
?savev2_actorrnnnetwork_input_mlp_dense_bias_read_readvariableopG
Csavev2_actorrnnnetwork_input_mlp_dense_kernel_1_read_readvariableopE
Asavev2_actorrnnnetwork_input_mlp_dense_bias_1_read_readvariableopB
>savev2_actorrnnnetwork_output_dense_kernel_read_readvariableop@
<savev2_actorrnnnetwork_output_dense_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop8savev2_actorrnnnetwork_action_kernel_read_readvariableop6savev2_actorrnnnetwork_action_bias_read_readvariableop@savev2_actorrnnnetwork_dynamic_unroll_kernel_read_readvariableopJsavev2_actorrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableop>savev2_actorrnnnetwork_dynamic_unroll_bias_read_readvariableopAsavev2_actorrnnnetwork_input_mlp_dense_kernel_read_readvariableop?savev2_actorrnnnetwork_input_mlp_dense_bias_read_readvariableopCsavev2_actorrnnnetwork_input_mlp_dense_kernel_1_read_readvariableopAsavev2_actorrnnnetwork_input_mlp_dense_bias_1_read_readvariableop>savev2_actorrnnnetwork_output_dense_kernel_read_readvariableop<savev2_actorrnnnetwork_output_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*|
_input_shapesk
i: : :2::
��:	(�:�:	�:�:
��:�:(2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
::&"
 
_output_shapes
:
��:%!

_output_shapes
:	(�:!

_output_shapes	
:�:%!

_output_shapes
:	�:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:$ 

_output_shapes

:(2: 

_output_shapes
:2:

_output_shapes
: 
\

__inference_<lambda>_161621*(
_construction_contextkEagerRuntime*
_input_shapes 
�
U
%__inference_get_initial_state_3405750

batch_size
identity

identity_1R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:���������(2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:���������(2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
U
%__inference_get_initial_state_3405101

batch_size
identity

identity_1R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������(2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������(2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:���������(2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:���������(2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
-
+__inference_function_with_signature_3405133�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *$
fR
__inference_<lambda>_1616212
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes "�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0/discount:0���������
>
0/observation-
action_0/observation:0���������
0
0/reward$
action_0/reward:0���������
6
0/step_type'
action_0/step_type:0���������
*
1/0#
action_1/0:0���������(
*
1/1#
action_1/1:0���������(:
action0
StatefulPartitionedCall:0���������;
state/00
StatefulPartitionedCall:1���������(;
state/10
StatefulPartitionedCall:2���������(tensorflow/serving/predict*�
get_initial_state�
2

batch_size$
get_initial_state_batch_size:0 -
0(
PartitionedCall:0���������(-
1(
PartitionedCall:1���������(tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:
�
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures

waction
xdistribution
yget_initial_state
zget_metadata
{get_train_step"
_generic_user_object
 "
trackable_list_wrapper
:	 (2Variable
 "
trackable_dict_wrapper
o
0
1
	2

3
4
5
6
7
8
9
10"
trackable_tuple_wrapper
5
0
1
2"
trackable_list_wrapper
`

|action
}get_initial_state
~get_train_step
get_metadata"
signature_map
/:-22ActorRnnNetwork/action/kernel
):'2ActorRnnNetwork/action/bias
9:7
��2%ActorRnnNetwork/dynamic_unroll/kernel
B:@	(�2/ActorRnnNetwork/dynamic_unroll/recurrent_kernel
2:0�2#ActorRnnNetwork/dynamic_unroll/bias
9:7	�2&ActorRnnNetwork/input_mlp/dense/kernel
3:1�2$ActorRnnNetwork/input_mlp/dense/bias
::8
��2&ActorRnnNetwork/input_mlp/dense/kernel
3:1�2$ActorRnnNetwork/input_mlp/dense/bias
5:3(22#ActorRnnNetwork/output/dense/kernel
/:-22!ActorRnnNetwork/output/dense/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
�
_state_spec
_flat_action_spec
_input_layers
_dynamic_unroll
_output_layers
_action_layers
regularization_losses
trainable_variables
 	variables
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
3
	state
1"
trackable_tuple_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
�
%cell
&regularization_losses
'trainable_variables
(	variables
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
.
*0
+1"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
	4

5
6
7
8
9
10"
trackable_list_wrapper
n
0
1
2
3
	4

5
6
7
8
9
10"
trackable_list_wrapper
�
regularization_losses
-metrics
.non_trainable_variables
/layer_metrics
0layer_regularization_losses
trainable_variables

1layers
 	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
2regularization_losses
3trainable_variables
4	variables
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
>
state_size

	kernel

recurrent_kernel
bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
�
&regularization_losses
Cmetrics
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses
'trainable_variables

Glayers
(	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
"0
#1
$2
3
*4
+5
,6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
2regularization_losses
Tmetrics
Unon_trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
3trainable_variables

Xlayers
4	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6regularization_losses
Ymetrics
Znon_trainable_variables
[layer_metrics
\layer_regularization_losses
7trainable_variables

]layers
8	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
:regularization_losses
^metrics
_non_trainable_variables
`layer_metrics
alayer_regularization_losses
;trainable_variables

blayers
<	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
�
?regularization_losses
cmetrics
dnon_trainable_variables
elayer_metrics
flayer_regularization_losses
@trainable_variables

glayers
A	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hregularization_losses
hmetrics
inon_trainable_variables
jlayer_metrics
klayer_regularization_losses
Itrainable_variables

llayers
J	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Lregularization_losses
mmetrics
nnon_trainable_variables
olayer_metrics
player_regularization_losses
Mtrainable_variables

qlayers
N	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Pregularization_losses
rmetrics
snon_trainable_variables
tlayer_metrics
ulayer_regularization_losses
Qtrainable_variables

vlayers
R	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference_action_3405345
__inference_action_3405552�
���
FullArgSpec8
args0�-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults�	
� 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_distribution_fn_3405734�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_get_initial_state_3405750�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_161621"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_161618"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6872139
0/discount0/observation0/reward0/step_type1/01/1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6872148
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6872156"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6872160"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�	
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�	
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecH
args@�=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecH
args@�=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 :
__inference_<lambda>_161618�

� 
� "� 	3
__inference_<lambda>_161621�

� 
� "� �
__inference_action_3405345�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
=�:
�
0���������(
�
1���������(

 
� "���

PolicyStep*
action �
action���������R
stateI�F
!�
state/0���������(
!�
state/1���������(
info� �
__inference_action_3405552�	
���
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������
W�T
(�%
policy_state/0���������(
(�%
policy_state/1���������(

 
� "���

PolicyStep*
action �
action���������R
stateI�F
!�
state/0���������(
!�
state/1���������(
info� �
#__inference_distribution_fn_3405734�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
=�:
�
0���������(
�
1���������(
� "���

PolicyStep�
action������
`
F�C

atol� 

loc����������

rtol� 
J�G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
�
j
parameters
� 
�
jname+tfp.distributions.Deterministic_ACTTypeSpecR
stateI�F
!�
state/0���������(
!�
state/1���������(
info� �
%__inference_get_initial_state_3405750c"�
�
�

batch_size 
� "=�:
�
0���������(
�
1���������(�
%__inference_signature_wrapper_6872139�	
���
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������
$
1/0�
1/0���������(
$
1/1�
1/1���������("���
*
action �
action���������
,
state/0!�
state/0���������(
,
state/1!�
state/1���������(�
%__inference_signature_wrapper_6872148{0�-
� 
&�#
!

batch_size�

batch_size "G�D
 
0�
0���������(
 
1�
1���������(Y
%__inference_signature_wrapper_68721560�

� 
� "�

int64�
int64 	=
%__inference_signature_wrapper_6872160�

� 
� "� 