Ты!
Ым
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
incompatible_shape_errorbool(Р
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
delete_old_dirsbool(И
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
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8µћ
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
Е
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*e
shared_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
ю
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	Р*
dtype0
э
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*c
shared_nameTRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
ц
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:Р*
dtype0
К
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р»*g
shared_nameXVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
Г
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel* 
_output_shapes
:
Р»*
dtype0
Б
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*e
shared_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
ъ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes	
:»*
dtype0
ш
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
»†*^
shared_nameOMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
с
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOpMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel* 
_output_shapes
:
»†*
dtype0
Л
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(†*h
shared_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
Д
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOpWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel*
_output_shapes
:	(†*
dtype0
п
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*\
shared_nameMKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
и
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:†*
dtype0
ё
CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
„
WActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpReadVariableOpCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias*
_output_shapes
:*
dtype0
ю
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*b
shared_nameSQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
ч
eActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOpQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel*
_output_shapes

:(*
dtype0
ц
OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
п
cActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpReadVariableOpOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias*
_output_shapes
:*
dtype0
ў
>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*O
shared_name@>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel
“
RValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes
:	Р*
dtype0
—
<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*M
shared_name><ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias
 
PValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias*
_output_shapes	
:Р*
dtype0
Џ
>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р»*O
shared_name@>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
”
RValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOp>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel* 
_output_shapes
:
Р»*
dtype0
—
<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*M
shared_name><ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
 
PValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOp<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias*
_output_shapes	
:»*
dtype0
ћ
7ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
»†*H
shared_name97ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
≈
KValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOpReadVariableOp7ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel* 
_output_shapes
:
»†*
dtype0
я
AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(†*R
shared_nameCAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
Ў
UValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel*
_output_shapes
:	(†*
dtype0
√
5ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*F
shared_name75ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
Љ
IValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOpReadVariableOp5ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias*
_output_shapes	
:†*
dtype0
Ш
ValueRnnNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*/
shared_name ValueRnnNetwork/dense_4/kernel
С
2ValueRnnNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOpValueRnnNetwork/dense_4/kernel*
_output_shapes

:(*
dtype0
Р
ValueRnnNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameValueRnnNetwork/dense_4/bias
Й
0ValueRnnNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOpValueRnnNetwork/dense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ЂT
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*жS
value№SBўS B“S
k
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures

actor_network_state
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
О
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18

0
1
2
 
 
ЧФ
VARIABLE_VALUETActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUERActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUEVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUETActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUEMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUEWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUEKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUEQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUEOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE7ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE5ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEValueRnnNetwork/dense_4/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEValueRnnNetwork/dense_4/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

 ref
 1

actor_network_state

	state
1
W
!_actor_network
"_policy_state_spec
#_policy_step_spec
$_value_network
Р
%_state_spec
&_lstm_encoder
'_projection_networks
(	variables
)trainable_variables
*regularization_losses
+	keras_api

,actor_network_state

	"state
"1
Т
-_state_spec
._lstm_encoder
/_postprocessing_layers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
 
Я
4_state_spec
5_input_encoder
6_lstm_network
7_output_encoder
8	variables
9trainable_variables
:regularization_losses
;	keras_api
z
<_means_projection_layer
	=_bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
F
0
	1

2
3
4
5
6
7
8
9
F
0
	1

2
3
4
5
6
7
8
9
 
≠
Bmetrics
Clayer_metrics
(	variables
)trainable_variables
Dnon_trainable_variables
*regularization_losses
Elayer_regularization_losses

Flayers
 
 
Я
G_state_spec
H_input_encoder
I_lstm_network
J_output_encoder
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

kernel
bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?
0
1
2
3
4
5
6
7
8
?
0
1
2
3
4
5
6
7
8
 
≠
Smetrics
Tlayer_metrics
0	variables
1trainable_variables
Unon_trainable_variables
2regularization_losses
Vlayer_regularization_losses

Wlayers
 
n
X_postprocessing_layers
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
\
]cell
^	variables
_trainable_variables
`regularization_losses
a	keras_api
 
1
0
	1

2
3
4
5
6
1
0
	1

2
3
4
5
6
 
≠
bmetrics
clayer_metrics
8	variables
9trainable_variables
dnon_trainable_variables
:regularization_losses
elayer_regularization_losses

flayers
h

kernel
bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
\
bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api

0
1
2

0
1
2
 
≠
ometrics
player_metrics
>	variables
?trainable_variables
qnon_trainable_variables
@regularization_losses
rlayer_regularization_losses

slayers
 
 
 
 

&0
'1
 
n
t_postprocessing_layers
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
\
ycell
z	variables
{trainable_variables
|regularization_losses
}	keras_api
 
1
0
1
2
3
4
5
6
1
0
1
2
3
4
5
6
 
∞
~metrics
layer_metrics
K	variables
Ltrainable_variables
Аnon_trainable_variables
Mregularization_losses
 Бlayer_regularization_losses
Вlayers

0
1

0
1
 
≤
Гmetrics
Дlayer_metrics
O	variables
Ptrainable_variables
Еnon_trainable_variables
Qregularization_losses
 Жlayer_regularization_losses
Зlayers
 
 
 
 

.0
/1

И0
Й1
К2

0
	1

2
3

0
	1

2
3
 
≤
Лmetrics
Мlayer_metrics
Y	variables
Ztrainable_variables
Нnon_trainable_variables
[regularization_losses
 Оlayer_regularization_losses
Пlayers
У
Р
state_size

kernel
recurrent_kernel
bias
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api

0
1
2

0
1
2
 
≤
Хmetrics
Цlayer_metrics
^	variables
_trainable_variables
Чnon_trainable_variables
`regularization_losses
 Шlayer_regularization_losses
Щlayers
 
 
 
 

50
61

0
1

0
1
 
≤
Ъmetrics
Ыlayer_metrics
g	variables
htrainable_variables
Ьnon_trainable_variables
iregularization_losses
 Эlayer_regularization_losses
Юlayers

0

0
 
≤
Яmetrics
†layer_metrics
k	variables
ltrainable_variables
°non_trainable_variables
mregularization_losses
 Ґlayer_regularization_losses
£layers
 
 
 
 

<0
=1

§0
•1
¶2

0
1
2
3

0
1
2
3
 
≤
Іmetrics
®layer_metrics
u	variables
vtrainable_variables
©non_trainable_variables
wregularization_losses
 ™layer_regularization_losses
Ђlayers
У
ђ
state_size

kernel
recurrent_kernel
bias
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api

0
1
2

0
1
2
 
≤
±metrics
≤layer_metrics
z	variables
{trainable_variables
≥non_trainable_variables
|regularization_losses
 іlayer_regularization_losses
µlayers
 
 
 
 

H0
I1
 
 
 
 
 
V
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
l

kernel
	bias
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api
l


kernel
bias
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
 
 
 
 

И0
Й1
К2
 

0
1
2

0
1
2
 
µ
¬metrics
√layer_metrics
С	variables
Тtrainable_variables
ƒnon_trainable_variables
Уregularization_losses
 ≈layer_regularization_losses
∆layers
 
 
 
 

]0
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
V
«	variables
»trainable_variables
…regularization_losses
 	keras_api
l

kernel
bias
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
l

kernel
bias
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
 
 
 
 

§0
•1
¶2
 

0
1
2

0
1
2
 
µ
”metrics
‘layer_metrics
≠	variables
Ѓtrainable_variables
’non_trainable_variables
ѓregularization_losses
 ÷layer_regularization_losses
„layers
 
 
 
 

y0
 
 
 
µ
Ўmetrics
ўlayer_metrics
ґ	variables
Јtrainable_variables
Џnon_trainable_variables
Єregularization_losses
 џlayer_regularization_losses
№layers

0
	1

0
	1
 
µ
Ёmetrics
ёlayer_metrics
Ї	variables
їtrainable_variables
яnon_trainable_variables
Љregularization_losses
 аlayer_regularization_losses
бlayers


0
1


0
1
 
µ
вmetrics
гlayer_metrics
Њ	variables
њtrainable_variables
дnon_trainable_variables
јregularization_losses
 еlayer_regularization_losses
жlayers
 
 
 
 
 
 
 
 
µ
зmetrics
иlayer_metrics
«	variables
»trainable_variables
йnon_trainable_variables
…regularization_losses
 кlayer_regularization_losses
лlayers

0
1

0
1
 
µ
мmetrics
нlayer_metrics
Ћ	variables
ћtrainable_variables
оnon_trainable_variables
Ќregularization_losses
 пlayer_regularization_losses
рlayers

0
1

0
1
 
µ
сmetrics
тlayer_metrics
ѕ	variables
–trainable_variables
уnon_trainable_variables
—regularization_losses
 фlayer_regularization_losses
хlayers
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
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
w
action_0/observationPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
j
action_0/rewardPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
action_0/step_typePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Б
action_1/actor_network_state/0Placeholder*'
_output_shapes
:€€€€€€€€€(*
dtype0*
shape:€€€€€€€€€(
Б
action_1/actor_network_state/1Placeholder*'
_output_shapes
:€€€€€€€€€(*
dtype0*
shape:€€€€€€€€€(
Ю

StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_typeaction_1/actor_network_state/0action_1/actor_network_state/1TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_384558
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€(:€€€€€€€€€(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_384567
Џ
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_384579
Х
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_384575
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOp_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpWActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpeActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpcActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpRValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpPValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpRValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpPValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpKValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOpUValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpIValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOp2ValueRnnNetwork/dense_4/kernel/Read/ReadVariableOp0ValueRnnNetwork/dense_4/bias/Read/ReadVariableOpConst*!
Tin
2	*
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
GPU 2J 8В *(
f#R!
__inference__traced_save_384673
„
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias7ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel5ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/biasValueRnnNetwork/dense_4/kernelValueRnnNetwork/dense_4/bias* 
Tin
2*
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_384743ѓ€
Е
ц
$__inference_signature_wrapper_384558
discount
observation

reward
	step_type
actor_network_state_0
actor_network_state_1
unknown:	Р
	unknown_0:	Р
	unknown_1:
Р»
	unknown_2:	»
	unknown_3:
»†
	unknown_4:	(†
	unknown_5:	†
	unknown_6:(
	unknown_7:
	unknown_8:
identity

identity_1

identity_2ИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *3
f.R,
*__inference_function_with_signature_3431652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_name0/observation:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:PL
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type:`\
'
_output_shapes
:€€€€€€€€€(
1
_user_specified_name1/actor_network_state/0:`\
'
_output_shapes
:€€€€€€€€€(
1
_user_specified_name1/actor_network_state/1
ј
Z
*__inference_function_with_signature_343225

batch_size
identity

identity_1Ї
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€(:€€€€€€€€€(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_get_initial_state_3432202
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2

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
°Ж
Ѓ
__inference_action_343550
	step_type

reward
discount
observation
actor_network_state_0
actor_network_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Р|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	РВ
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
Р»~
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	»Г
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
»†Д
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(†
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	†{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:(x
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpҐeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpҐgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpҐhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpҐUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpҐaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpҐ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpF
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
strided_slice/stack_2в
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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
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
:€€€€€€€€€2
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
€€€€€€€€€2
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
concat_2/axisЗ
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
:€€€€€€€€€2	
ReshapeЛ
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2С

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_1J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axisХ
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axisХ
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axisЙ
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
	Reshape_1П

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_2С

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_3“
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim™
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims÷
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim™
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1≈
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeЫ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeґ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape€
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeш
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЄ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulч
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЇ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd√
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu€
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЊ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulэ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp¬
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd…
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu§
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2о
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceа
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1ђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ш
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis 
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatя
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:€€€€€€€€€»2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape¬
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yѕ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask№
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta—
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeБ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisн
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat™
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€»2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose†
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeВ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackЖ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2М
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceГ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permЛ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yИ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulн
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yГ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessр
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Я
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedн
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstС
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosо
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yО
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulс
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yЛ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessф
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1•
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedс
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstЩ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Ћ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:€€€€€€€€€»*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeezeћ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1°
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectІ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1В
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
»†*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЄ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulЗ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(†*
dtype02j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2[
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1∞
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€†2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addА
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype02i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpљ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddЖ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2b
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimГ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(*
	num_split2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split“
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul…
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhҐ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1°
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2»
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1¶
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2р
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim¶
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€(2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims≤
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*'
_output_shapes
:€€€€€€€€€(*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeо
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpЦ
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulн
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp≠
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd„
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeа
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeф
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanhї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x≠
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЃ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addь
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like…
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpр
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddџ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeЏ
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1В
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusЪ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЃ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeб
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2≥
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape÷
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2Ѕ
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2л	
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceєActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0«ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2ї
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceх
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackЅActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2љ
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1µ
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2є
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisз
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2єActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0√ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0њActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2і
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatН
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2°
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackЪ
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1С
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2 
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ІActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Ы
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice√
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeЊ
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЋ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1¬
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ѕ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceґ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs°ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsх
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Constі
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosз
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesд
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroђ
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeе
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstЙ
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2я
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackН
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Э
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSlice„ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0еActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceъ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorй
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 2”
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackС
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1С
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2ѓ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1StridedSliceбActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensor:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1ь
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2№
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0А
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2ё
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1©
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsдActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs¶
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgsџActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0бActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1ј
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2„
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЫ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Constњ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ъ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2™
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSliceЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0АActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice≤
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2ч
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0ґ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2щ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Х
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs€ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0ъActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsП	
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityцActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2т
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЎ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shapeѓ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const”
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ƒ
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2ы	
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceЉActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape:output:0 ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceл
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1≥
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackџ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1џ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2З

љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSliceЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1∆
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2Ѕ
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0 
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2√
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1љ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgs…ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsЇ
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgsјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0∆ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1п
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2Љ
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeї
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constп
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeь
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis№
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЩ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastToBroadcastTo;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTom
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
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape–
Deterministic_1/sample/ShapeShapedActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/ConstҐ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_sliceХ
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0Щ
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1№
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgsЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0У
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis≤
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЫ
"Deterministic_1/sample/BroadcastToBroadcastTodActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1¶
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack™
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1™
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2ф
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1О
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisК
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1‘
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
clip_by_value/Minimum/yґ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yР
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityє

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_1є

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_2…
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 2ћ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2 
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2ќ
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2“
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2‘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2Ѓ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2∆
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2ƒ
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_nameobservation:^Z
'
_output_shapes
:€€€€€€€€€(
/
_user_specified_nameactor_network_state/0:^Z
'
_output_shapes
:€€€€€€€€€(
/
_user_specified_nameactor_network_state/1
Ж
d
$__inference_signature_wrapper_384575
unknown:	 
identity	ИҐStatefulPartitionedCall≥
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
GPU 2J 8В *3
f.R,
*__inference_function_with_signature_3432412
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
г
,
*__inference_function_with_signature_343252х
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
GPU 2J 8В *!
fR
__inference_<lambda>_5072
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
п
&
$__inference_signature_wrapper_384579З
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
GPU 2J 8В *3
f.R,
*__inference_function_with_signature_3432522
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
ъ
j
*__inference_function_with_signature_343241
unknown:	 
identity	ИҐStatefulPartitionedCall°
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
GPU 2J 8В *!
fR
__inference_<lambda>_5042
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
ч?
Е
__inference__traced_save_384673
file_prefix'
#savev2_variable_read_readvariableop	s
osavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopq
msavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias_read_readvariableopu
qsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableops
osavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableopl
hsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableopv
rsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableopj
fsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableopb
^savev2_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopp
lsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopn
jsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableop]
Ysavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernel_read_readvariableop[
Wsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_bias_read_readvariableop]
Ysavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableop[
Wsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableopV
Rsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableop`
\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopT
Psavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableop=
9savev2_valuernnnetwork_dense_4_kernel_read_readvariableop;
7savev2_valuernnnetwork_dense_4_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameї
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ќ
value√BјB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names≤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЧ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableoposavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopmsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias_read_readvariableopqsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableoposavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableophsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableoprsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableopfsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableop^savev2_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableoplsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopjsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopYsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopWsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_bias_read_readvariableopYsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableopWsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableopRsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableop\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopPsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableop9savev2_valuernnnetwork_dense_4_kernel_read_readvariableop7savev2_valuernnnetwork_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*«
_input_shapesµ
≤: : :	Р:Р:
Р»:»:
»†:	(†:†::(::	Р:Р:
Р»:»:
»†:	(†:†:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	Р:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Р»:!

_output_shapes	
:»:&"
 
_output_shapes
:
»†:%!

_output_shapes
:	(†:!

_output_shapes	
:†: 	

_output_shapes
::$
 

_output_shapes

:(: 

_output_shapes
::%!

_output_shapes
:	Р:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Р»:!

_output_shapes	
:»:&"
 
_output_shapes
:
»†:%!

_output_shapes
:	(†:!

_output_shapes	
:†:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
≥
T
$__inference_get_initial_state_344127

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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

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
Ш
_
__inference_<lambda>_504!
readvariableop_resource:	 
identity	ИҐReadVariableOpp
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
ъ
ь
*__inference_function_with_signature_343165
	step_type

reward
discount
observation
actor_network_state_0
actor_network_state_1
unknown:	Р
	unknown_0:	Р
	unknown_1:
Р»
	unknown_2:	»
	unknown_3:
»†
	unknown_4:	(†
	unknown_5:	†
	unknown_6:(
	unknown_7:
	unknown_8:
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_action_3431382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_name0/observation:`\
'
_output_shapes
:€€€€€€€€€(
1
_user_specified_name1/actor_network_state/0:`\
'
_output_shapes
:€€€€€€€€€(
1
_user_specified_name1/actor_network_state/1
ј
T
$__inference_signature_wrapper_384567

batch_size
identity

identity_1ј
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€(:€€€€€€€€€(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *3
f.R,
*__inference_function_with_signature_3432252
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2

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
≥
T
$__inference_get_initial_state_343220

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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

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
Тб
„
"__inference_distribution_fn_344111
	step_type

reward
discount
observation
actor_network_state_0
actor_network_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Р|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	РВ
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
Р»~
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	»Г
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
»†Д
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(†
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	†{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:(x
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4ИҐdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpҐeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpҐgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpҐhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpҐUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpҐaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpҐ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpF
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
strided_slice/stack_2в
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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
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
:€€€€€€€€€2
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
€€€€€€€€€2
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
concat_2/axisЗ
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
:€€€€€€€€€2	
ReshapeЛ
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2С

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_1J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axisХ
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axisХ
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axisЙ
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
	Reshape_1П

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_2С

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_3“
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim™
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims÷
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim™
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1≈
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeЫ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeґ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape€
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeш
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЄ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulч
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЇ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd√
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu€
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЊ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulэ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp¬
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd…
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu§
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2о
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceа
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1ђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ш
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis 
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatя
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:€€€€€€€€€»2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape¬
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yѕ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask№
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta—
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeБ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisн
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat™
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€»2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose†
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeВ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackЖ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2М
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceГ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permЛ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yИ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulн
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yГ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessр
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Я
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedн
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstС
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosо
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yО
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulс
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yЛ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessф
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1•
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedс
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstЩ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Ћ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:€€€€€€€€€»*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeezeћ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1°
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectІ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1В
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
»†*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЄ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulЗ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(†*
dtype02j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2[
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1∞
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€†2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addА
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype02i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpљ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddЖ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2b
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimГ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(*
	num_split2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split“
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul…
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhҐ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1°
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2»
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1¶
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2р
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim¶
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€(2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims≤
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*'
_output_shapes
:€€€€€€€€€(*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeо
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpЦ
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulн
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp≠
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd„
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeа
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeф
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanhї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x≠
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЃ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addь
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like…
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpр
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddџ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeЏ
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1В
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusЪ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЃ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeб
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2≥
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape÷
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2Ѕ
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2л	
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceєActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0«ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2ї
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceх
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackЅActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2љ
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1µ
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2є
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisз
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2єActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0√ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0њActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2і
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatН
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2°
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackЪ
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1С
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2 
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ІActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Ы
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice√
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeЊ
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЋ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1¬
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ѕ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceґ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs°ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsх
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Constі
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosз
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesд
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroђ
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeе
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstЙ
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2я
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackН
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Э
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSlice„ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0еActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceъ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorй
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 2”
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackС
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1С
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2ѓ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1StridedSliceбActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensor:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1ь
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2№
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0А
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2ё
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1©
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsдActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs¶
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgsџActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0бActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1ј
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2„
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЫ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Constњ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ъ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2™
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSliceЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0АActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice≤
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2ч
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0ґ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2щ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Х
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs€ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0ъActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsП	
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityцActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2т
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЎ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shapeѓ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const”
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ƒ
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2ы	
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceЉActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape:output:0 ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceл
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1≥
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackџ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1џ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2З

љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSliceЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1∆
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2Ѕ
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0 
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2√
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1љ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgs…ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsЇ
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgsјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0∆ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1п
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2Љ
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeї
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constп
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeь
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis№
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЩ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastToBroadcastTo;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTom
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

Identity√

Identity_1IdentitydActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1i

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_2є

Identity_3IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_3є

Identity_4IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_4…
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 2ћ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2 
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2ќ
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2“
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2‘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2Ѓ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2∆
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2ƒ
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_nameobservation:^Z
'
_output_shapes
:€€€€€€€€€(
/
_user_specified_nameactor_network_state/0:^Z
'
_output_shapes
:€€€€€€€€€(
/
_user_specified_nameactor_network_state/1
Y

__inference_<lambda>_507*(
_construction_contextkEagerRuntime*
_input_shapes 
™e
ф
"__inference__traced_restore_384743
file_prefix#
assignvariableop_variable:	 z
gassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel:	Рt
eassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias:	Р}
iassignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel:
Р»v
gassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias:	»t
`assignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel:
»†}
jassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel:	(†m
^assignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias:	†d
Vassignvariableop_8_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias:v
dassignvariableop_9_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel:(q
cassignvariableop_10_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias:e
Rassignvariableop_11_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernel:	Р_
Passignvariableop_12_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_bias:	Рf
Rassignvariableop_13_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel:
Р»_
Passignvariableop_14_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias:	»_
Kassignvariableop_15_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel:
»†h
Uassignvariableop_16_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel:	(†X
Iassignvariableop_17_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias:	†D
2assignvariableop_18_valuernnnetwork_dense_4_kernel:(>
0assignvariableop_19_valuernnnetwork_dense_4_bias:
identity_21ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ќ
value√BјB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1м
AssignVariableOp_1AssignVariableOpgassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2к
AssignVariableOp_2AssignVariableOpeassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3о
AssignVariableOp_3AssignVariableOpiassignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4м
AssignVariableOp_4AssignVariableOpgassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5е
AssignVariableOp_5AssignVariableOp`assignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6п
AssignVariableOp_6AssignVariableOpjassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOp^assignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8џ
AssignVariableOp_8AssignVariableOpVassignvariableop_8_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9й
AssignVariableOp_9AssignVariableOpdassignvariableop_9_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10л
AssignVariableOp_10AssignVariableOpcassignvariableop_10_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Џ
AssignVariableOp_11AssignVariableOpRassignvariableop_11_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ў
AssignVariableOp_12AssignVariableOpPassignvariableop_12_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Џ
AssignVariableOp_13AssignVariableOpRassignvariableop_13_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ў
AssignVariableOp_14AssignVariableOpPassignvariableop_14_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15”
AssignVariableOp_15AssignVariableOpKassignvariableop_15_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ё
AssignVariableOp_16AssignVariableOpUassignvariableop_16_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17—
AssignVariableOp_17AssignVariableOpIassignvariableop_17_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ї
AssignVariableOp_18AssignVariableOp2assignvariableop_18_valuernnnetwork_dense_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Є
AssignVariableOp_19AssignVariableOp0assignvariableop_19_valuernnnetwork_dense_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20f
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_21ю
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
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
€Е
¶
__inference_action_343138
	time_step
time_step_1
time_step_2
time_step_3
policy_state
policy_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Р|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	РВ
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
Р»~
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	»Г
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
»†Д
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(†
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	†{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:(x
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpҐeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpҐgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpҐhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpҐUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpҐaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpҐ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpI
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
strided_slice/stack_2в
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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
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
:€€€€€€€€€2
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
€€€€€€€€€2
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
concat_2/axisЗ
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
:€€€€€€€€€2	
ReshapeВ
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2К

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_1M
Shape_2Shapetime_step_2*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axisХ
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axisХ
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axisЙ
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
	Reshape_1П

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_2С

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_3“
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim™
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims÷
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim™
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1≈
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeЫ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeґ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape€
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeш
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЄ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulч
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЇ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd√
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu€
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЊ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulэ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp¬
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd…
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu§
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2о
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceа
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1ђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ш
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis 
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatя
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:€€€€€€€€€»2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape¬
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yѕ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask№
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta—
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeБ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisн
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat™
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€»2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose†
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeВ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackЖ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2М
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceГ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permЛ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yИ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulн
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yГ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessр
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Я
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedн
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstС
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosо
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yО
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulс
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yЛ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessф
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1•
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedс
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstЩ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Ћ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:€€€€€€€€€»*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeezeћ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1°
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectІ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1В
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
»†*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЄ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulЗ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(†*
dtype02j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2[
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1∞
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€†2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addА
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype02i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpљ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddЖ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2b
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimГ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(*
	num_split2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split“
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul…
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhҐ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1°
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2»
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1¶
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2р
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim¶
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€(2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims≤
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*'
_output_shapes
:€€€€€€€€€(*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeо
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpЦ
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulн
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp≠
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd„
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeа
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeф
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanhї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x≠
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЃ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addь
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like…
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpр
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddџ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeЏ
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1В
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusЪ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЃ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeб
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2≥
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape÷
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2Ѕ
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2л	
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceєActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0«ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2ї
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceх
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackЅActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2љ
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1µ
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2є
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisз
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2єActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0√ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0њActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2і
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatН
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2°
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackЪ
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1С
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2 
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ІActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Ы
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice√
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeЊ
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЋ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1¬
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ѕ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceґ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs°ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsх
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Constі
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosз
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesд
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroђ
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeе
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstЙ
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2я
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackН
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Э
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSlice„ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0еActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceъ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorй
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 2”
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackС
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1С
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2ѓ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1StridedSliceбActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensor:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1ь
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2№
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0А
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2ё
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1©
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsдActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs¶
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgsџActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0бActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1ј
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2„
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЫ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Constњ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ъ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2™
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSliceЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0АActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice≤
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2ч
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0ґ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2щ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Х
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs€ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0ъActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsП	
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityцActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2т
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЎ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shapeѓ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const”
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ƒ
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2ы	
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceЉActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape:output:0 ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceл
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1≥
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackџ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1џ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2З

љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSliceЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1∆
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2Ѕ
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0 
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2√
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1љ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgs…ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsЇ
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgsјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0∆ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1п
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2Љ
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeї
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constп
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeь
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis№
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЩ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastToBroadcastTo;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTom
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
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape–
Deterministic_1/sample/ShapeShapedActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/ConstҐ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_sliceХ
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0Щ
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1№
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgsЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0У
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis≤
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЫ
"Deterministic_1/sample/BroadcastToBroadcastTodActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1¶
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack™
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1™
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2ф
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1О
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisК
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1‘
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
clip_by_value/Minimum/yґ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yР
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityє

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_1є

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_2…
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 2ћ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2 
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2ќ
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2“
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2‘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2Ѓ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2∆
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2ƒ
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:UQ
'
_output_shapes
:€€€€€€€€€(
&
_user_specified_namepolicy_state:UQ
'
_output_shapes
:€€€€€€€€€(
&
_user_specified_namepolicy_state
ыЗ
р
__inference_action_343843
time_step_step_type
time_step_reward
time_step_discount
time_step_observation&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Р|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	РВ
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
Р»~
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	»Г
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
»†Д
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	(†
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	†{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:(x
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpҐeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpҐgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpҐfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpҐhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpҐUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpҐaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpҐ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpP
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
strided_slice/stack_2в
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
concat/axisЛ
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
:€€€€€€€€€(2
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
concat_1/axisУ
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
:€€€€€€€€€(2	
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
:€€€€€€€€€2
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
€€€€€€€€€2
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
concat_2/axisЗ
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
:€€€€€€€€€2	
ReshapeШ
SelectV2SelectV2Reshape:output:0zeros:output:0"policy_state_actor_network_state_0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2Ю

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0"policy_state_actor_network_state_1*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_1T
Shape_2Shapetime_step_discount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axisХ
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:(2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axisХ
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yr
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axisЙ
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
	Reshape_1П

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_2С

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

SelectV2_3“
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimі
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims÷
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimі
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1≈
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeЫ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeґ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape€
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeш
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЄ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulч
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЇ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd√
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu€
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЊ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulэ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp¬
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd…
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu§
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2о
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceа
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape®
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1ђ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ш
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis 
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatя
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*,
_output_shapes
:€€€€€€€€€»2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape¬
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yѕ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask№
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startк
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta—
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeБ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisн
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat™
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€»2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose†
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeВ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackЖ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2М
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceГ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permЛ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1к
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yИ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulн
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yГ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessр
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Я
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedн
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstС
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosо
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yО
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulс
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yЛ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessф
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1•
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedс
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstЩ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Ћ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:€€€€€€€€€»*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeezeћ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1°
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectІ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1В
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
»†*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЄ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulЗ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	(†*
dtype02j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2[
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1∞
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€†2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addА
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype02i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpљ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddЖ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2b
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimГ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(:€€€€€€€€€(*
	num_split2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split“
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2Z
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul…
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€(2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhҐ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1°
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1÷
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€(2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2»
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€(2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1¶
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€(2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2р
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim¶
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€(2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims≤
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*'
_output_shapes
:€€€€€€€€€(*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeо
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpЦ
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulн
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp≠
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd„
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeа
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeф
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanhї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x≠
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulї
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЃ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addь
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like…
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpр
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddџ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeЏ
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1В
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusЪ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЃ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeб
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2≥
∞ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape÷
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2Ѕ
ЊActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1—
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2√
јActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2л	
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceєActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0«ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0…ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2ї
ЄActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceх
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackЅActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2љ
ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1µ
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2є
ґActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisз
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2єActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0√ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0њActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2і
±ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatН
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2°
ЮActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackЪ
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1С
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2£
†ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2 
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ІActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0©ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Ы
ШActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice√
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeЊ
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЋ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1¬
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ѕ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ГActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceґ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs°ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsх
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Constі
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosз
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesд
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroђ
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeе
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2—
ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstЙ
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2я
№ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackН
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Э
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSlice„ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0еActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceъ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensorй
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 2”
–ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Const_1Н
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2б
ёActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stackС
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1С
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2г
аActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2ѓ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1StridedSliceбActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/shape_as_tensor:output:0зActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1ь
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2№
ўActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0А
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2ё
џActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1©
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsдActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ў
÷ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs¶
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgsџActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0бActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2џ
ЎActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1ј
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2„
‘ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЫ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Constњ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ъ
чActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1√
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ь
щActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2™
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_sliceStridedSliceЁActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0АActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_1:output:0ВActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice≤
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2ч
фActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0ґ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2щ
цActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1Х
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs€ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0ъActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2ф
сActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsП	
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityцActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2т
пActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЎ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shapeѓ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2ґ
≥ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const”
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2ƒ
ЅActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2ы	
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceЉActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape:output:0 ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_sliceл
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1≥
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Const_1„
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2∆
√ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackџ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1џ
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2»
≈ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2З

љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSliceЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0ћActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0ќActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1∆
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2Ѕ
ЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0 
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2√
јActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1љ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgs…ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsЇ
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgsјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0∆ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:2ј
љActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1п
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:2Љ
єActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeї
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2Є
µActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constп
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityЊActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2Њ
їActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeь
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis№
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2¬ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ƒActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЩ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastToBroadcastTo;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTom
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
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape–
Deterministic_1/sample/ShapeShapedActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/ConstҐ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_sliceХ
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0Щ
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1№
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgsЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0У
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis≤
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЫ
"Deterministic_1/sample/BroadcastToBroadcastTodActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1¶
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack™
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1™
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2ф
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1О
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisК
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1‘
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
clip_by_value/Minimum/yґ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yР
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityє

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_1є

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity_2…
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€(:€€€€€€€€€(: : : : : : : : : : 2ћ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2 
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2ќ
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2“
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2–
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2‘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2Ѓ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2∆
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2ƒ
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:X T
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nametime_step/observation:kg
'
_output_shapes
:€€€€€€€€€(
<
_user_specified_name$"policy_state/actor_network_state/0:kg
'
_output_shapes
:€€€€€€€€€(
<
_user_specified_name$"policy_state/actor_network_state/1"®L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Н
actionВ
4

0/discount&
action_0/discount:0€€€€€€€€€
>
0/observation-
action_0/observation:0€€€€€€€€€
0
0/reward$
action_0/reward:0€€€€€€€€€
6
0/step_type'
action_0/step_type:0€€€€€€€€€
R
1/actor_network_state/07
 action_1/actor_network_state/0:0€€€€€€€€€(
R
1/actor_network_state/17
 action_1/actor_network_state/1:0€€€€€€€€€(:
action0
StatefulPartitionedCall:0€€€€€€€€€O
state/actor_network_state/00
StatefulPartitionedCall:1€€€€€€€€€(O
state/actor_network_state/10
StatefulPartitionedCall:2€€€€€€€€€(tensorflow/serving/predict*м
get_initial_state÷
2

batch_size$
get_initial_state_batch_size:0 A
actor_network_state/0(
PartitionedCall:0€€€€€€€€€(A
actor_network_state/1(
PartitionedCall:1€€€€€€€€€(tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:€Ж
й
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures
цaction
чdistribution
шget_initial_state
щget_metadata
ъget_train_step"
_generic_user_object
9
actor_network_state"
trackable_dict_wrapper
:	 (2Variable
 "
trackable_dict_wrapper
ѓ
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18"
trackable_tuple_wrapper
5
0
1
2"
trackable_list_wrapper
d
ыaction
ьget_initial_state
эget_train_step
юget_metadata"
signature_map
 "
trackable_list_wrapper
g:e	Р2TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
a:_Р2RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
j:h
Р»2VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
c:a»2TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
a:_
»†2MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
j:h	(†2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
Z:X†2KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
Q:O2CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
c:a(2QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
]:[2OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
Q:O	Р2>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel
K:IР2<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias
R:P
Р»2>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
K:I»2<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
K:I
»†27ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
T:R	(†2AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
D:B†25ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
0:.(2ValueRnnNetwork/dense_4/kernel
*:(2ValueRnnNetwork/dense_4/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
 ref
 1"
trackable_tuple_wrapper
9
actor_network_state"
trackable_dict_wrapper
3
	state
1"
trackable_tuple_wrapper
u
!_actor_network
"_policy_state_spec
#_policy_step_spec
$_value_network"
_generic_user_object
е
%_state_spec
&_lstm_encoder
'_projection_networks
(	variables
)trainable_variables
*regularization_losses
+	keras_api
€__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_layer
9
,actor_network_state"
trackable_dict_wrapper
3
	"state
"1"
trackable_tuple_wrapper
з
-_state_spec
._lstm_encoder
/_postprocessing_layers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
ф
4_state_spec
5_input_encoder
6_lstm_network
7_output_encoder
8	variables
9trainable_variables
:regularization_losses
;	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
ѕ
<_means_projection_layer
	=_bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Bmetrics
Clayer_metrics
(	variables
)trainable_variables
Dnon_trainable_variables
*regularization_losses
Elayer_regularization_losses

Flayers
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ф
G_state_spec
H_input_encoder
I_lstm_network
J_output_encoder
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Smetrics
Tlayer_metrics
0	variables
1trainable_variables
Unon_trainable_variables
2regularization_losses
Vlayer_regularization_losses

Wlayers
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
√
X_postprocessing_layers
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
±
]cell
^	variables
_trainable_variables
`regularization_losses
a	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
bmetrics
clayer_metrics
8	variables
9trainable_variables
dnon_trainable_variables
:regularization_losses
elayer_regularization_losses

flayers
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
љ

kernel
bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
±
bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
ometrics
player_metrics
>	variables
?trainable_variables
qnon_trainable_variables
@regularization_losses
rlayer_regularization_losses

slayers
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
√
t_postprocessing_layers
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
±
ycell
z	variables
{trainable_variables
|regularization_losses
}	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
≥
~metrics
layer_metrics
K	variables
Ltrainable_variables
Аnon_trainable_variables
Mregularization_losses
 Бlayer_regularization_losses
Вlayers
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Гmetrics
Дlayer_metrics
O	variables
Ptrainable_variables
Еnon_trainable_variables
Qregularization_losses
 Жlayer_regularization_losses
Зlayers
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
8
И0
Й1
К2"
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Лmetrics
Мlayer_metrics
Y	variables
Ztrainable_variables
Нnon_trainable_variables
[regularization_losses
 Оlayer_regularization_losses
Пlayers
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
и
Р
state_size

kernel
recurrent_kernel
bias
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Хmetrics
Цlayer_metrics
^	variables
_trainable_variables
Чnon_trainable_variables
`regularization_losses
 Шlayer_regularization_losses
Щlayers
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ъmetrics
Ыlayer_metrics
g	variables
htrainable_variables
Ьnon_trainable_variables
iregularization_losses
 Эlayer_regularization_losses
Юlayers
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Яmetrics
†layer_metrics
k	variables
ltrainable_variables
°non_trainable_variables
mregularization_losses
 Ґlayer_regularization_losses
£layers
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
8
§0
•1
¶2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Іmetrics
®layer_metrics
u	variables
vtrainable_variables
©non_trainable_variables
wregularization_losses
 ™layer_regularization_losses
Ђlayers
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
и
ђ
state_size

kernel
recurrent_kernel
bias
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±metrics
≤layer_metrics
z	variables
{trainable_variables
≥non_trainable_variables
|regularization_losses
 іlayer_regularization_losses
µlayers
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
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
Ђ
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

kernel
	bias
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ


kernel
bias
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
И0
Й1
К2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¬metrics
√layer_metrics
С	variables
Тtrainable_variables
ƒnon_trainable_variables
Уregularization_losses
 ≈layer_regularization_losses
∆layers
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
]0"
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
Ђ
«	variables
»trainable_variables
…regularization_losses
 	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

kernel
bias
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
£__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

kernel
bias
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
§0
•1
¶2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
”metrics
‘layer_metrics
≠	variables
Ѓtrainable_variables
’non_trainable_variables
ѓregularization_losses
 ÷layer_regularization_losses
„layers
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ўmetrics
ўlayer_metrics
ґ	variables
Јtrainable_variables
Џnon_trainable_variables
Єregularization_losses
 џlayer_regularization_losses
№layers
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ёmetrics
ёlayer_metrics
Ї	variables
їtrainable_variables
яnon_trainable_variables
Љregularization_losses
 аlayer_regularization_losses
бlayers
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
вmetrics
гlayer_metrics
Њ	variables
њtrainable_variables
дnon_trainable_variables
јregularization_losses
 еlayer_regularization_losses
жlayers
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
 "
trackable_list_wrapper
Є
зmetrics
иlayer_metrics
«	variables
»trainable_variables
йnon_trainable_variables
…regularization_losses
 кlayer_regularization_losses
лlayers
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
мmetrics
нlayer_metrics
Ћ	variables
ћtrainable_variables
оnon_trainable_variables
Ќregularization_losses
 пlayer_regularization_losses
рlayers
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
сmetrics
тlayer_metrics
ѕ	variables
–trainable_variables
уnon_trainable_variables
—regularization_losses
 фlayer_regularization_losses
хlayers
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
Г2А
__inference_action_343550
__inference_action_343843«
Њ≤Ї
FullArgSpec8
args0Ъ-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsҐ	
Ґ 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
"__inference_distribution_fn_344111Ђ
§≤†
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
$__inference_get_initial_state_344127¶
Э≤Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЃBЂ
__inference_<lambda>_507"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЃBЂ
__inference_<lambda>_504"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
§B°
$__inference_signature_wrapper_384558
0/discount0/observation0/reward0/step_type1/actor_network_state/01/actor_network_state/1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќBЋ
$__inference_signature_wrapper_384567
batch_size"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
јBљ
$__inference_signature_wrapper_384575"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
јBљ
$__inference_signature_wrapper_384579"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—ќ
≈≤Ѕ
FullArgSpec?
args7Ъ4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—ќ
≈≤Ѕ
FullArgSpec?
args7Ъ4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я№
”≤ѕ
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ	
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ёџ
“≤ќ
FullArgSpecH
args@Ъ=
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
defaultsЪ

 

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ёџ
“≤ќ
FullArgSpecH
args@Ъ=
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
defaultsЪ

 

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ёџ
“≤ќ
FullArgSpecH
args@Ъ=
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
defaultsЪ

 

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ёџ
“≤ќ
FullArgSpecH
args@Ъ=
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
defaultsЪ

 

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2ЅЊ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2ЅЊ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2ЅЊ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2ЅЊ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 7
__inference_<lambda>_504Ґ

Ґ 
™ "К 	0
__inference_<lambda>_507Ґ

Ґ 
™ "™ ц
__inference_action_343550Ў
	
бҐЁ
’Ґ—
∆≤¬
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€4
observation%К"
observation€€€€€€€€€
Б™~
|
actor_network_stateeЪb
/К,
actor_network_state/0€€€€€€€€€(
/К,
actor_network_state/1€€€€€€€€€(

 
™ "е≤б

PolicyStep*
action К
action€€€€€€€€€Щ
stateП™Л
И
actor_network_stateqЪn
5К2
state/actor_network_state/0€€€€€€€€€(
5К2
state/actor_network_state/1€€€€€€€€€(
infoҐ Ї
__inference_action_343843Ь
	
•Ґ°
ЩҐХ
о≤к
TimeStep6
	step_type)К&
time_step/step_type€€€€€€€€€0
reward&К#
time_step/reward€€€€€€€€€4
discount(К%
time_step/discount€€€€€€€€€>
observation/К,
time_step/observation€€€€€€€€€
Э™Щ
Ц
actor_network_stateЪ|
<К9
"policy_state/actor_network_state/0€€€€€€€€€(
<К9
"policy_state/actor_network_state/1€€€€€€€€€(

 
™ "е≤б

PolicyStep*
action К
action€€€€€€€€€Щ
stateП™Л
И
actor_network_stateqЪn
5К2
state/actor_network_state/0€€€€€€€€€(
5К2
state/actor_network_state/1€€€€€€€€€(
infoҐ о
"__inference_distribution_fn_344111«
	
ЁҐў
—ҐЌ
∆≤¬
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€4
observation%К"
observation€€€€€€€€€
Б™~
|
actor_network_stateeЪb
/К,
actor_network_state/0€€€€€€€€€(
/К,
actor_network_state/1€€€€€€€€€(
™ "Ў≤‘

PolicyStepЬ
actionСТНЅҐљ
`
F™C

atolК 

locК€€€€€€€€€

rtolК 
J™G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
Ґ
j
parameters
Ґ 
Ґ
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpecЩ
stateП™Л
И
actor_network_stateqЪn
5К2
state/actor_network_state/0€€€€€€€€€(
5К2
state/actor_network_state/1€€€€€€€€€(
infoҐ —
$__inference_get_initial_state_344127®"Ґ
Ґ
К

batch_size 
™ "Б™~
|
actor_network_stateeЪb
/К,
actor_network_state/0€€€€€€€€€(
/К,
actor_network_state/1€€€€€€€€€(Л
$__inference_signature_wrapper_384558в
	
фҐр
Ґ 
и™д
.

0/discount К

0/discount€€€€€€€€€
8
0/observation'К$
0/observation€€€€€€€€€
*
0/rewardК
0/reward€€€€€€€€€
0
0/step_type!К
0/step_type€€€€€€€€€
L
1/actor_network_state/01К.
1/actor_network_state/0€€€€€€€€€(
L
1/actor_network_state/11К.
1/actor_network_state/1€€€€€€€€€("№™Ў
*
action К
action€€€€€€€€€
T
state/actor_network_state/05К2
state/actor_network_state/0€€€€€€€€€(
T
state/actor_network_state/15К2
state/actor_network_state/1€€€€€€€€€(ц
$__inference_signature_wrapper_384567Ќ0Ґ-
Ґ 
&™#
!

batch_sizeК

batch_size "Ш™Ф
H
actor_network_state/0/К,
actor_network_state/0€€€€€€€€€(
H
actor_network_state/1/К,
actor_network_state/1€€€€€€€€€(X
$__inference_signature_wrapper_3845750Ґ

Ґ 
™ "™

int64К
int64 	<
$__inference_signature_wrapper_384579Ґ

Ґ 
™ "™ 