ЛО
ѓ"Ш"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
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
delete_old_dirsbool(
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
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisintџџџџџџџџџ"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
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
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ЫЮ

"agent/ValueRnnNetwork/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"agent/ValueRnnNetwork/dense_5/bias

6agent/ValueRnnNetwork/dense_5/bias/Read/ReadVariableOpReadVariableOp"agent/ValueRnnNetwork/dense_5/bias*
_output_shapes
:*
dtype0
Є
$agent/ValueRnnNetwork/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*5
shared_name&$agent/ValueRnnNetwork/dense_5/kernel

8agent/ValueRnnNetwork/dense_5/kernel/Read/ReadVariableOpReadVariableOp$agent/ValueRnnNetwork/dense_5/kernel*
_output_shapes

:(*
dtype0
Я
;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
Ш
Oagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOpReadVariableOp;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias*
_output_shapes	
: *
dtype0
ы
Gagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	( *X
shared_nameIGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
ф
[agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel*
_output_shapes
:	( *
dtype0
з
=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *N
shared_name?=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
а
Qagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOpReadVariableOp=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel*
_output_shapes
:	  *
dtype0
м
Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias
е
Vagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOpBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias*
_output_shapes
: *
dtype0
ф
Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *U
shared_nameFDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel
н
Xagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOpDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel*
_output_shapes

:  *
dtype0
м
Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
е
Vagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOpBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias*
_output_shapes
: *
dtype0
ф
Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *U
shared_nameFDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
н
Xagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOpDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel*
_output_shapes

:M *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
ь
Jagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
х
^agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes
:*
dtype0
ѕ
Lagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*]
shared_nameNLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
ю
`agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel*
_output_shapes
:	*
dtype0
ћ
Qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
є
eagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:*
dtype0

]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*n
shared_name_]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel

qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOp]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel* 
_output_shapes
:
*
dtype0

Sagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *d
shared_nameUSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
ќ
gagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOpSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel*
_output_shapes
:	 *
dtype0

Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *k
shared_name\Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias

nagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOpZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias*
_output_shapes
: *
dtype0

\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *m
shared_name^\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel

pagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes

:  *
dtype0

Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *k
shared_name\Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias

nagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
: *
dtype0

\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *m
shared_name^\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel

pagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:M *
dtype0
r
conv2d/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias_1
k
!conv2d/bias_1/Read/ReadVariableOpReadVariableOpconv2d/bias_1*
_output_shapes
:*
dtype0

conv2d/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel_1
{
#conv2d/kernel_1/Read/ReadVariableOpReadVariableOpconv2d/kernel_1*&
_output_shapes
:*
dtype0
p
dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias_1
i
 dense/bias_1/Read/ReadVariableOpReadVariableOpdense/bias_1*
_output_shapes
:*
dtype0
x
dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel_1
q
"dense/kernel_1/Read/ReadVariableOpReadVariableOpdense/kernel_1*
_output_shapes

:*
dtype0
j
global_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	

NoOpNoOp
г
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ыв
valueРвBМв BДв
с
collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
	get_initial_state

get_metadata
get_train_step

signatures*

observation
1* 

actor_network_state* 
JD
VARIABLE_VALUEglobal_step%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ъ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25*
s
)_time_step_spec
*_policy_state_spec
+_policy_step_spec
,_trajectory_spec
-_wrapped_policy*

.trace_0
/trace_1* 

0trace_0* 

1trace_0* 
* 
* 
K

2action
3get_initial_state
4get_train_step
5get_metadata* 
* 
* 
TN
VARIABLE_VALUEdense/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv2d/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ
VARIABLE_VALUE\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
Ё
VARIABLE_VALUEZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ
VARIABLE_VALUE\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
Ё
VARIABLE_VALUEZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
Є
VARIABLE_VALUE]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUE
dense/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv2d/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE$agent/ValueRnnNetwork/dense_5/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE"agent/ValueRnnNetwork/dense_5/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*

observation
3* 

actor_network_state* 

	*state
*1* 

observation
1* 

6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network*
* 
* 
* 
* 
* 
* 
* 
* 
ц
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_input_tensor_spec
C_state_spec
D_lstm_encoder
E_projection_networks*

Fobservation
F3* 

Gactor_network_state* 

	8state
81* 

Fobservation
F1* 
ш
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_input_tensor_spec
O_state_spec
P_lstm_encoder
Q_postprocessing_layers*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
* 
ѕ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_input_tensor_spec
^_state_spec
__input_encoder
`_lstm_network
a_output_encoder*
Ї
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_projection_layer*
* 
* 
b
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12*
b
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
* 
ѕ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_input_tensor_spec
u_state_spec
v_input_encoder
w_lstm_network
x_output_encoder*
І
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

'kernel
(bias*
* 

D0
E1*
* 
* 
* 
R
0
1
2
3
4
5
6
7
8
9
10*
R
0
1
2
3
4
5
6
7
8
9
10*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_input_tensor_spec
_preprocessing_nest
_flat_preprocessing_layers
_preprocessing_combiner
_postprocessing_layers*
Ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	cell*
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
Ќ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

kernel
bias*
* 

P0
Q1*
* 
* 
* 
R
0
1
2
3
 4
!5
"6
#7
$8
%9
&10*
R
0
1
2
3
 4
!5
"6
#7
$8
%9
&10*
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
* 
* 
* 
* 
Ѕ
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
Ќ_input_tensor_spec
­_preprocessing_nest
Ў_flat_preprocessing_layers
Џ_preprocessing_combiner
А_postprocessing_layers*
Ё
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
	Зcell*
* 

'0
(1*

'0
(1*
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 

_0
`1*
* 
* 
* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

Т0
У1*

Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses* 

Ъ0
Ы1
Ь2*

0
1
2*

0
1
2*
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
ы
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
и_random_generator
й
state_size

kernel
recurrent_kernel
bias*
* 

h0*
* 
* 
* 

0
1*

0
1*
* 

кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 

v0
w1*
* 
* 
* 
<
0
1
2
3
 4
!5
"6
#7*
<
0
1
2
3
 4
!5
"6
#7*
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses*
* 
* 
* 
* 

ф0
х1*

ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses* 

ь0
э1
ю2*

$0
%1
&2*

$0
%1
&2*
* 

яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*
* 
* 
ы
є	variables
ѕtrainable_variables
іregularization_losses
ї	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses
њ_random_generator
ћ
state_size

$kernel
%recurrent_kernel
&bias*
* 
* 
* 
* 
* 
* 
4
Т0
У1
2
Ъ3
Ы4
Ь5*
* 
* 
* 
Э
ќlayer-0
§layer_with_weights-0
§layer-1
ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
щ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses* 
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ќ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
Ќ
	variables
 trainable_variables
Ёregularization_losses
Ђ	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses

kernel
bias*
* 

0*
* 
* 
* 

0
1
2*

0
1
2*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
ф0
х1
Џ2
ь3
э4
ю5*
* 
* 
* 
Э
Њlayer-0
Ћlayer_with_weights-0
Ћlayer-1
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses*
щ
Вlayer-0
Гlayer_with_weights-0
Гlayer-1
Дlayer-2
Еlayer-3
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses*
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses* 
* 
* 

С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses* 
Ќ
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses

 kernel
!bias*
Ќ
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

"kernel
#bias*
* 

З0*
* 
* 
* 

$0
%1
&2*

$0
%1
&2*
* 

гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
є	variables
ѕtrainable_variables
іregularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses*
* 
* 
* 
* 

и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses* 
Ќ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
щtrace_0
ъtrace_1
ыtrace_2
ьtrace_3* 
:
эtrace_0
юtrace_1
яtrace_2
№trace_3* 

ё	variables
ђtrainable_variables
ѓregularization_losses
є	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses* 
Я
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses

kernel
bias
!§_jit_compiled_convolution_op*

ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
	variables
 trainable_variables
Ёregularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses* 
Ќ
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*
:
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_3* 
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 

П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses* 
Я
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses

kernel
bias
!Ы_jit_compiled_convolution_op*

Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses* 

в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*
:
нtrace_0
оtrace_1
пtrace_2
рtrace_3* 
:
сtrace_0
тtrace_1
уtrace_2
фtrace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 
* 
* 

 0
!1*

 0
!1*
* 

ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*
* 
* 

"0
#1*

"0
#1*
* 

яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses* 

љtrace_0
њtrace_1* 

ћtrace_0
ќtrace_1* 

0
1*

0
1*
* 

§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

ќ0
§1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ё	variables
ђtrainable_variables
ѓregularization_losses
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

 trace_0* 

Ёtrace_0* 
* 
$
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses* 

Їtrace_0
Јtrace_1* 

Љtrace_0
Њtrace_1* 

0
1*

0
1*
* 

Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

Аtrace_0* 

Бtrace_0* 
* 

Њ0
Ћ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses* 

Зtrace_0
Иtrace_1* 

Йtrace_0
Кtrace_1* 

0
1*

0
1*
* 

Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

Рtrace_0* 

Сtrace_0* 
* 
* 
* 
* 

Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses* 

Чtrace_0* 

Шtrace_0* 
* 
* 
* 

Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses* 

Юtrace_0* 

Яtrace_0* 
* 
$
В0
Г1
Д2
Е3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
l
action_0_discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0_observation_directionPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0_observation_imagePlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
j
action_0_rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0_step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_1_actor_network_state_0Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_1_actor_network_state_1Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
ч

StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_directionaction_0_observation_imageaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1dense/kernel_1dense/bias_1conv2d/kernel_1conv2d/bias_1\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9445795
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
М
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9445804
л
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9445816

StatefulPartitionedCall_1StatefulPartitionedCallglobal_step*
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9445812
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp"dense/kernel_1/Read/ReadVariableOp dense/bias_1/Read/ReadVariableOp#conv2d/kernel_1/Read/ReadVariableOp!conv2d/bias_1/Read/ReadVariableOppagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpnagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOppagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpnagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpgagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpqagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpeagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOp`agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOp^agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOpXagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpVagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpXagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpVagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpQagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOp[agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpOagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOp8agent/ValueRnnNetwork/dense_5/kernel/Read/ReadVariableOp6agent/ValueRnnNetwork/dense_5/bias/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_9446936
ы
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepdense/kernel_1dense/bias_1conv2d/kernel_1conv2d/bias_1\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasdense/kernel
dense/biasconv2d/kernelconv2d/biasDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernelBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias$agent/ValueRnnNetwork/dense_5/kernel"agent/ValueRnnNetwork/dense_5/bias*'
Tin 
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_9447027мя
Ф
Я
G__inference_sequential_layer_call_and_return_conditional_losses_9445995

inputs(
conv2d_9445974:
conv2d_9445976:
identityЂconv2d/StatefulPartitionedCallН
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445961
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9445974conv2d_9445976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445992o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
k
+__inference_function_with_signature_1944102
unknown:	 
identity	ЂStatefulPartitionedCall
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
GPU 2J 8 *!
fR
__inference_<lambda>_919^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
ж
е
G__inference_sequential_layer_call_and_return_conditional_losses_9446093
lambda_input(
conv2d_9446085:
conv2d_9446087:
identityЂconv2d/StatefulPartitionedCallУ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445961
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446085conv2d_9446087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445992o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
бx
Ђ
#__inference__traced_restore_9447027
file_prefix&
assignvariableop_global_step:	 3
!assignvariableop_1_dense_kernel_1:-
assignvariableop_2_dense_bias_1:<
"assignvariableop_3_conv2d_kernel_1:.
 assignvariableop_4_conv2d_bias_1:
oassignvariableop_5_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel:M {
massignvariableop_6_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias: 
oassignvariableop_7_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel:  {
massignvariableop_8_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias: y
fassignvariableop_9_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel:	 
qassignvariableop_10_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel:
t
eassignvariableop_11_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias:	s
`assignvariableop_12_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:	l
^assignvariableop_13_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:2
 assignvariableop_14_dense_kernel:,
assignvariableop_15_dense_bias:;
!assignvariableop_16_conv2d_kernel:-
assignvariableop_17_conv2d_bias:j
Xassignvariableop_18_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel:M d
Vassignvariableop_19_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias: j
Xassignvariableop_20_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel:  d
Vassignvariableop_21_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias: d
Qassignvariableop_22_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel:	  n
[assignvariableop_23_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel:	( ^
Oassignvariableop_24_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias:	 J
8assignvariableop_25_agent_valuernnnetwork_dense_5_kernel:(D
6assignvariableop_26_agent_valuernnnetwork_dense_5_bias:
identity_28ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9№

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ћ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_global_stepIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_5AssignVariableOpoassignvariableop_5_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_6AssignVariableOpmassignvariableop_6_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_7AssignVariableOpoassignvariableop_7_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_8AssignVariableOpmassignvariableop_8_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_9AssignVariableOpfassignvariableop_9_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_10AssignVariableOpqassignvariableop_10_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_11AssignVariableOpeassignvariableop_11_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_12AssignVariableOp`assignvariableop_12_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_13AssignVariableOp^assignvariableop_13_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv2d_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_conv2d_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_18AssignVariableOpXassignvariableop_18_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_19AssignVariableOpVassignvariableop_19_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_20AssignVariableOpXassignvariableop_20_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_21AssignVariableOpVassignvariableop_21_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_22AssignVariableOpQassignvariableop_22_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_23AssignVariableOp[assignvariableop_23_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_24AssignVariableOpOassignvariableop_24_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_25AssignVariableOp8assignvariableop_25_agent_valuernnnetwork_dense_5_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_26AssignVariableOp6assignvariableop_26_agent_valuernnnetwork_dense_5_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ё
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
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
Ф
`
D__inference_flatten_layer_call_and_return_conditional_losses_9445992

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџHX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф
Ѓ
.__inference_sequential_1_layer_call_fn_9445927
lambda_1_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445911o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input

Ю
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445937
lambda_1_input
dense_9445931:
dense_9445933:
identityЂdense/StatefulPartitionedCallС
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445833
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445931dense_9445933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445845u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
ЋH
ы
 __inference__traced_save_9446936
file_prefix*
&savev2_global_step_read_readvariableop	-
)savev2_dense_kernel_1_read_readvariableop+
'savev2_dense_bias_1_read_readvariableop.
*savev2_conv2d_kernel_1_read_readvariableop,
(savev2_conv2d_bias_1_read_readvariableop{
wsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopy
usavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableop{
wsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopy
usavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias_read_readvariableopr
nsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableop|
xsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableopp
lsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableopk
gsavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopi
esavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableopc
_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableopa
]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableopc
_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel_read_readvariableopa
]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias_read_readvariableop\
Xsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableopf
bsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopZ
Vsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableopC
?savev2_agent_valuernnnetwork_dense_5_kernel_read_readvariableopA
=savev2_agent_valuernnnetwork_dense_5_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: э

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop)savev2_dense_kernel_1_read_readvariableop'savev2_dense_bias_1_read_readvariableop*savev2_conv2d_kernel_1_read_readvariableop(savev2_conv2d_bias_1_read_readvariableopwsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopusavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableopwsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopusavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias_read_readvariableopnsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableopxsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableoplsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableopgsavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopesavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableop]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableop_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel_read_readvariableop]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias_read_readvariableopXsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableopbsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopVsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableop?savev2_agent_valuernnnetwork_dense_5_kernel_read_readvariableop=savev2_agent_valuernnnetwork_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesѕ
ђ: : :::::M : :  : :	 :
::	::::::M : :  : :	  :	( : :(:: 2(
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

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:M : 

_output_shapes
: :$ 

_output_shapes

:  : 	

_output_shapes
: :%
!

_output_shapes
:	 :&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:M : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	  :%!

_output_shapes
:	( :!

_output_shapes	
: :$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
­
E
)__inference_flatten_layer_call_fn_9446814

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446281`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_9446686

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445833

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х	
ѓ
B__inference_dense_layer_call_and_return_conditional_losses_9446134

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446614

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К

I__inference_sequential_1_layer_call_and_return_conditional_losses_9446444

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
Я
G__inference_sequential_layer_call_and_return_conditional_losses_9446065

inputs(
conv2d_9446057:
conv2d_9446059:
identityЂconv2d/StatefulPartitionedCallН
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446039
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446057conv2d_9446059*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445992o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446696

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

.__inference_sequential_1_layer_call_fn_9446503

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ч
__inference_action_1944352
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:

pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpЂoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpЂqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
ShapeShapediscount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:џџџџџџџџџ?
Shape_2Shapediscount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ю
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :і
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Д
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   е
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	О
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         у
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЈ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aј
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџА
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
І
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ї
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЁ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   н
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Е
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskЅ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ћ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:Ї
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:е
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ а
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:Є
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Е
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ш
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Т
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџј
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ј
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 б
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџе
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0н
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџи
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitї
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџЩ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџШ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџю
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :в
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ№
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
ѕ
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ћ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџђ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџж
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB В
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ё
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџЌ

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЃ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2т
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2ц
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2д
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2А
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2Ў
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameobservation/direction:b^
/
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/image:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/1
ђ
Ї
,__inference_sequential_layer_call_fn_9446002
lambda_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
О

'__inference_dense_layer_call_fn_9446633

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445845o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К

I__inference_sequential_1_layer_call_and_return_conditional_losses_9446528

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
Я
G__inference_sequential_layer_call_and_return_conditional_losses_9446284

inputs(
conv2d_9446263:
conv2d_9446265:
identityЂconv2d/StatefulPartitionedCallН
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446250
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446263conv2d_9446265*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446281o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

.__inference_sequential_1_layer_call_fn_9446512

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х

G__inference_sequential_layer_call_and_return_conditional_losses_9446578

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџe

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446780

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К

I__inference_sequential_1_layer_call_and_return_conditional_losses_9446544

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О

'__inference_dense_layer_call_fn_9446746

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
Ё
,__inference_sequential_layer_call_fn_9446553

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_lambda_1_layer_call_fn_9446604

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445887`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
D
(__inference_lambda_layer_call_fn_9446766

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446328h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш

(__inference_conv2d_layer_call_fn_9446676

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446250

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х	
ѓ
B__inference_dense_layer_call_and_return_conditional_losses_9446643

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х	
ѓ
B__inference_dense_layer_call_and_return_conditional_losses_9445845

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
+__inference_function_with_signature_1944019
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:M 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity	

identity_1

identity_2ЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_directionobservation_imageactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_action_1943986k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_name0/observation/direction:d`
/
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/image:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/1
ђ
Ї
,__inference_sequential_layer_call_fn_9446291
lambda_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
Ф
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446281

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџHX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф
Ѓ
.__inference_sequential_1_layer_call_fn_9446148
lambda_1_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
К

I__inference_sequential_1_layer_call_and_return_conditional_losses_9446428

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
Ё
,__inference_sequential_layer_call_fn_9446462

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446809

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446820

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџHX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446624

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_lambda_1_layer_call_fn_9446717

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446176`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

U
%__inference_get_initial_state_1944813

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџW
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџ[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
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
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446737

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­
E
)__inference_flatten_layer_call_fn_9446701

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445992`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

.__inference_sequential_1_layer_call_fn_9446403

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445887

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
Ї
,__inference_sequential_layer_call_fn_9446081
lambda_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
ђ
_
__inference_<lambda>_919!
readvariableop_resource:	 
identity	ЂReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
ф
Ѓ
.__inference_sequential_1_layer_call_fn_9446216
lambda_1_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
Ї

ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_9446799

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
е
G__inference_sequential_layer_call_and_return_conditional_losses_9446105
lambda_input(
conv2d_9446097:
conv2d_9446099:
identityЂconv2d/StatefulPartitionedCallУ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446039
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446097conv2d_9446099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445992o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
Ї

ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_9445973

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

.__inference_sequential_1_layer_call_fn_9446412

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445911o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш
Я
__inference_action_1943986
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
policy_state
policy_state_1
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:

pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpЂoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpЂqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
ShapeShapetime_step_2*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџw
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*(
_output_shapes
:џџџџџџџџџ}

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*(
_output_shapes
:џџџџџџџџџB
Shape_2Shapetime_step_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ф
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :№
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_4QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Д
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   е
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	О
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         у
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЈ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aј
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџА
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
І
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ї
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЁ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   н
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Е
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskЅ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ћ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:Ї
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:е
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ а
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:Є
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Е
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ш
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Т
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџј
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ј
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 б
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџе
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0н
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџи
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitї
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџЩ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџШ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџю
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :в
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ№
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
ѕ
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ћ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџђ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџж
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB В
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ё
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџЌ

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЃ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2т
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2ц
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2д
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2А
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2Ў
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:ZV
/
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state
я

Ц
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445852

inputs
dense_9445846:
dense_9445848:
identityЂdense/StatefulPartitionedCallЙ
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445833
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445846dense_9445848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445845u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446773

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х

G__inference_sequential_layer_call_and_return_conditional_losses_9446494

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџe

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
D
(__inference_lambda_layer_call_fn_9446648

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445961h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ю
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446236
lambda_1_input
dense_9446230:
dense_9446232:
identityЂdense/StatefulPartitionedCallС
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446176
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446230dense_9446232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
о
'
%__inference_signature_wrapper_9445816ѕ
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
GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1944113*(
_construction_contextkEagerRuntime*
_input_shapes 
Й
C
'__inference_re_lu_layer_call_fn_9446691

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445984h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446176

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х

G__inference_sequential_layer_call_and_return_conditional_losses_9446478

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџe

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

Ц
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446141

inputs
dense_9446135:
dense_9446137:
identityЂdense/StatefulPartitionedCallЙ
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446122
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446135dense_9446137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
D
(__inference_lambda_layer_call_fn_9446653

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446039h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

Ц
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446200

inputs
dense_9446194:
dense_9446196:
identityЂdense/StatefulPartitionedCallЙ
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446176
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446194dense_9446196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446707

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџHX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш

(__inference_conv2d_layer_call_fn_9446789

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
C
'__inference_re_lu_layer_call_fn_9446804

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ю
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446226
lambda_1_input
dense_9446220:
dense_9446222:
identityЂdense/StatefulPartitionedCallС
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446122
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446220dense_9446222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input

[
+__inference_function_with_signature_1944086

batch_size
identity

identity_1Њ
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_get_initial_state_1944081a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџc

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"
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

ё
%__inference_signature_wrapper_9445795
discount
observation_direction
observation_image

reward
	step_type
actor_network_state_0
actor_network_state_1
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:M 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity	

identity_1

identity_2ЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_directionobservation_imageactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1944019k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_name0/observation/direction:d`
/
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/image:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/1
я

Ц
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445911

inputs
dense_9445905:
dense_9445907:
identityЂdense/StatefulPartitionedCallЙ
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445887
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445905dense_9445907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445845u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
Я
G__inference_sequential_layer_call_and_return_conditional_losses_9446354

inputs(
conv2d_9446346:
conv2d_9446348:
identityЂconv2d/StatefulPartitionedCallН
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446328
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446346conv2d_9446348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446281o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
е
G__inference_sequential_layer_call_and_return_conditional_losses_9446382
lambda_input(
conv2d_9446374:
conv2d_9446376:
identityЂconv2d/StatefulPartitionedCallУ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446250
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446374conv2d_9446376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446281o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446727

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ю
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445947
lambda_1_input
dense_9445941:
dense_9445943:
identityЂdense/StatefulPartitionedCallС
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445887
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445941dense_9445943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445845u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
Л
D
(__inference_lambda_layer_call_fn_9446761

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446250h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б
-
+__inference_function_with_signature_1944113т
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
GPU 2J 8 *!
fR
__inference_<lambda>_922*(
_construction_contextkEagerRuntime*
_input_shapes 
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446039

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9445961

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
Ё
,__inference_sequential_layer_call_fn_9446453

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Г
__inference_action_1944587
time_step_step_type
time_step_reward
time_step_discount#
time_step_observation_direction
time_step_observation_image&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:

pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpЂoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpЂqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
ShapeShapetime_step_discount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : c
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
SelectV2SelectV2Reshape:output:0zeros:output:0"policy_state_actor_network_state_0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0"policy_state_actor_network_state_1*
T0*(
_output_shapes
:џџџџџџџџџI
Shape_2Shapetime_step_discount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : g
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ј
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ь
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Д
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   е
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	О
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         у
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЈ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aј
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџА
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
І
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ї
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЁ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   н
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Е
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskЅ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ћ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:Ї
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:е
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ а
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:Є
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Е
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ш
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Т
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџј
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ј
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 б
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџе
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0н
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџи
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitї
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџЩ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџШ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџю
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :в
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ№
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
ѕ
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ћ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџђ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџж
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB В
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ё
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџЌ

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЃ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2т
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2ц
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2д
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2А
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2Ў
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:hd
'
_output_shapes
:џџџџџџџџџ
9
_user_specified_name!time_step/observation/direction:lh
/
_output_shapes
:џџџџџџџџџ
5
_user_specified_nametime_step/observation/image:lh
(
_output_shapes
:џџџџџџџџџ
<
_user_specified_name$"policy_state/actor_network_state/0:lh
(
_output_shapes
:џџџџџџџџџ
<
_user_specified_name$"policy_state/actor_network_state/1
Х

G__inference_sequential_layer_call_and_return_conditional_losses_9446594

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџe

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Y

__inference_<lambda>_922*(
_construction_contextkEagerRuntime*
_input_shapes 
ф
Ѓ
.__inference_sequential_1_layer_call_fn_9445859
lambda_1_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_1_input
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446328

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

U
%__inference_signature_wrapper_9445804

batch_size
identity

identity_1А
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1944086a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџc

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"
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
ђ
Ї
,__inference_sequential_layer_call_fn_9446370
lambda_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input

U
%__inference_get_initial_state_1944081

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџW
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџ[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
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
ёі

#__inference_distribution_fn_1944797
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:

pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1	

identity_2	

identity_3

identity_4ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpЂoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpЂqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpЂpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
ShapeShapediscount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:џџџџџџџџџ?
Shape_2Shapediscount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ю
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :і
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Д
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   е
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	О
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         у
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЈ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aј
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџА
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
І
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   ї
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЁ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   н
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Е
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskџ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	З
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Й
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskЅ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ћ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:Ї
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:е
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ а
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:Є
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Е
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ш
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Т
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџј
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ј
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 б
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџе
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0н
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџи
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitї
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџЩ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџШ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџљ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџю
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :в
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ№
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
ѕ
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ћ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџђ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџж
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0	*
_output_shapes
: Е

Identity_1IdentityhCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: Ќ

Identity_3IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ

Identity_4IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЃ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2т
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2ц
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2ф
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2д
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2А
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2Ў
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameobservation/direction:b^
/
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/image:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/1

F
*__inference_lambda_1_layer_call_fn_9446599

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445833`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446660

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
Ё
,__inference_sequential_layer_call_fn_9446562

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
е
G__inference_sequential_layer_call_and_return_conditional_losses_9446394
lambda_input(
conv2d_9446386:
conv2d_9446388:
identityЂconv2d/StatefulPartitionedCallУ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446328
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446386conv2d_9446388*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446262м
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446273Я
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446281o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelambda_input
Х	
ѓ
B__inference_dense_layer_call_and_return_conditional_losses_9446756

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й
e
%__inference_signature_wrapper_9445812
unknown:	 
identity	ЂStatefulPartitionedCall
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
GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1944102^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
њ
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446122

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446667

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_lambda_1_layer_call_fn_9446712

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446122`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ПL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѕ
actionъ
4

0/discount&
action_0_discount:0џџџџџџџџџ
R
0/observation/direction7
 action_0_observation_direction:0џџџџџџџџџ
R
0/observation/image;
action_0_observation_image:0џџџџџџџџџ
0
0/reward$
action_0_reward:0џџџџџџџџџ
6
0/step_type'
action_0_step_type:0џџџџџџџџџ
S
1/actor_network_state/08
 action_1_actor_network_state_0:0џџџџџџџџџ
S
1/actor_network_state/18
 action_1_actor_network_state_1:0џџџџџџџџџ6
action,
StatefulPartitionedCall:0	џџџџџџџџџP
state/actor_network_state/01
StatefulPartitionedCall:1џџџџџџџџџP
state/actor_network_state/11
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict*ю
get_initial_stateи
2

batch_size$
get_initial_state_batch_size:0 B
actor_network_state/0)
PartitionedCall:0џџџџџџџџџB
actor_network_state/1)
PartitionedCall:1џџџџџџџџџtensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:аТ
ћ
collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
	get_initial_state

get_metadata
get_train_step

signatures"
_generic_user_object
9
observation
1"
trackable_tuple_wrapper
9
actor_network_state"
trackable_dict_wrapper
:	 (2global_step
 "
trackable_dict_wrapper
ч
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25"
trackable_tuple_wrapper

)_time_step_spec
*_policy_state_spec
+_policy_step_spec
,_trajectory_spec
-_wrapped_policy"
trackable_dict_wrapper
Й
.trace_0
/trace_12
__inference_action_1944352
__inference_action_1944587Ч
ОВК
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z.trace_0z/trace_1
№
0trace_02г
#__inference_distribution_fn_1944797Ћ
ЄВ 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z0trace_0
э
1trace_02а
%__inference_get_initial_state_1944813І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z1trace_0
ЎBЋ
__inference_<lambda>_922"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЎBЋ
__inference_<lambda>_919"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
`

2action
3get_initial_state
4get_train_step
5get_metadata"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:2dense/kernel
:2
dense/bias
':%2conv2d/kernel
:2conv2d/bias
n:lM 2\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
h:f 2Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
n:l  2\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel
h:f 2Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias
f:d	 2Sagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
q:o
2]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
`:^2Qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
_:]	2Lagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
X:V2Jagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
:2dense/kernel
:2
dense/bias
':%2conv2d/kernel
:2conv2d/bias
V:TM 2Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
P:N 2Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
V:T  2Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel
P:N 2Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias
P:N	  2=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
Z:X	( 2Gagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
J:H 2;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
6:4(2$agent/ValueRnnNetwork/dense_5/kernel
0:.2"agent/ValueRnnNetwork/dense_5/bias
9
observation
3"
trackable_tuple_wrapper
9
actor_network_state"
trackable_dict_wrapper
3
	*state
*1"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper
 
6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network"
_generic_user_object
рBн
__inference_action_1944352	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"Ч
ОВК
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЌBЉ
__inference_action_1944587time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/directiontime_step/observation/image"policy_state/actor_network_state/0"policy_state/actor_network_state/1"Ч
ОВК
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЭBЪ
#__inference_distribution_fn_1944797	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"Ћ
ЄВ 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
сBо
%__inference_get_initial_state_1944813
batch_size"І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ФBС
%__inference_signature_wrapper_9445795
0/discount0/observation/direction0/observation/image0/reward0/step_type1/actor_network_state/01/actor_network_state/1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЯBЬ
%__inference_signature_wrapper_9445804
batch_size"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
СBО
%__inference_signature_wrapper_9445812"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
СBО
%__inference_signature_wrapper_9445816"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_input_tensor_spec
C_state_spec
D_lstm_encoder
E_projection_networks"
_tf_keras_layer
9
Fobservation
F3"
trackable_tuple_wrapper
9
Gactor_network_state"
trackable_dict_wrapper
3
	8state
81"
trackable_tuple_wrapper
9
Fobservation
F1"
trackable_tuple_wrapper
§
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_input_tensor_spec
O_state_spec
P_lstm_encoder
Q_postprocessing_layers"
_tf_keras_layer
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_input_tensor_spec
^_state_spec
__input_encoder
`_lstm_network
a_output_encoder"
_tf_keras_layer
М
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_projection_layer"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12"
trackable_list_wrapper
~
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_input_tensor_spec
u_state_spec
v_input_encoder
w_lstm_network
x_output_encoder"
_tf_keras_layer
Л
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
Б
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_input_tensor_spec
_preprocessing_nest
_flat_preprocessing_layers
_preprocessing_combiner
_postprocessing_layers"
_tf_keras_layer
Ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	cell"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
д2бЮ
ХВС
FullArgSpec?
args74
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
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2бЮ
ХВС
FullArgSpec?
args74
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
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
С
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
 4
!5
"6
#7
$8
%9
&10"
trackable_list_wrapper
n
0
1
2
3
 4
!5
"6
#7
$8
%9
&10"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
К
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
Ќ_input_tensor_spec
­_preprocessing_nest
Ў_flat_preprocessing_layers
Џ_preprocessing_combiner
А_postprocessing_layers"
_tf_keras_layer
Ж
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
	Зcell"
_tf_keras_layer
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0
Т0
У1"
trackable_list_wrapper
Ћ
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
8
Ъ0
Ы1
Ь2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
с2ол
вВЮ
FullArgSpecH
args@=
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
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
FullArgSpecH
args@=
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
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 

в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
и_random_generator
й
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
 4
!5
"6
#7"
trackable_list_wrapper
X
0
1
2
3
 4
!5
"6
#7"
trackable_list_wrapper
 "
trackable_list_wrapper
И
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0
ф0
х1"
trackable_list_wrapper
Ћ
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_layer
8
ь0
э1
ю2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
с2ол
вВЮ
FullArgSpecH
args@=
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
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
FullArgSpecH
args@=
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
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 

є	variables
ѕtrainable_variables
іregularization_losses
ї	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses
њ_random_generator
ћ
state_size

$kernel
%recurrent_kernel
&bias"
_tf_keras_layer
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
P
Т0
У1
2
Ъ3
Ы4
Ь5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ч
ќlayer-0
§layer_with_weights-0
§layer-1
ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential

layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
С
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
С
	variables
 trainable_variables
Ёregularization_losses
Ђ	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
"
_generic_user_object
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
trackable_dict_wrapper
 "
trackable_list_wrapper
P
ф0
х1
Џ2
ь3
э4
ю5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ч
Њlayer-0
Ћlayer_with_weights-0
Ћlayer-1
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_sequential

Вlayer-0
Гlayer_with_weights-0
Гlayer-1
Дlayer-2
Еlayer-3
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ћ
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
С
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
С
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
З0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
є	variables
ѕtrainable_variables
іregularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
"
_generic_user_object
 "
trackable_list_wrapper
Ћ
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
С
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
і
щtrace_0
ъtrace_1
ыtrace_2
ьtrace_32
.__inference_sequential_1_layer_call_fn_9445859
.__inference_sequential_1_layer_call_fn_9446403
.__inference_sequential_1_layer_call_fn_9446412
.__inference_sequential_1_layer_call_fn_9445927Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zщtrace_0zъtrace_1zыtrace_2zьtrace_3
т
эtrace_0
юtrace_1
яtrace_2
№trace_32я
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446428
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446444
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445937
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445947Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zэtrace_0zюtrace_1zяtrace_2z№trace_3
Ћ
ё	variables
ђtrainable_variables
ѓregularization_losses
є	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses

kernel
bias
!§_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ю
trace_0
trace_1
trace_2
trace_32ћ
,__inference_sequential_layer_call_fn_9446002
,__inference_sequential_layer_call_fn_9446453
,__inference_sequential_layer_call_fn_9446462
,__inference_sequential_layer_call_fn_9446081Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
к
trace_0
trace_1
trace_2
trace_32ч
G__inference_sequential_layer_call_and_return_conditional_losses_9446478
G__inference_sequential_layer_call_and_return_conditional_losses_9446494
G__inference_sequential_layer_call_and_return_conditional_losses_9446093
G__inference_sequential_layer_call_and_return_conditional_losses_9446105Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
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
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
	variables
 trainable_variables
Ёregularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
Ћ
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
С
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
і
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_32
.__inference_sequential_1_layer_call_fn_9446148
.__inference_sequential_1_layer_call_fn_9446503
.__inference_sequential_1_layer_call_fn_9446512
.__inference_sequential_1_layer_call_fn_9446216Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЗtrace_0zИtrace_1zЙtrace_2zКtrace_3
т
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32я
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446528
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446544
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446226
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446236Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
Ћ
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses

kernel
bias
!Ы_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
ю
нtrace_0
оtrace_1
пtrace_2
рtrace_32ћ
,__inference_sequential_layer_call_fn_9446291
,__inference_sequential_layer_call_fn_9446553
,__inference_sequential_layer_call_fn_9446562
,__inference_sequential_layer_call_fn_9446370Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zнtrace_0zоtrace_1zпtrace_2zрtrace_3
к
сtrace_0
тtrace_1
уtrace_2
фtrace_32ч
G__inference_sequential_layer_call_and_return_conditional_losses_9446578
G__inference_sequential_layer_call_and_return_conditional_losses_9446594
G__inference_sequential_layer_call_and_return_conditional_losses_9446382
G__inference_sequential_layer_call_and_return_conditional_losses_9446394Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zсtrace_0zтtrace_1zуtrace_2zфtrace_3
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
И
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
И
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
ж
љtrace_0
њtrace_12
*__inference_lambda_1_layer_call_fn_9446599
*__inference_lambda_1_layer_call_fn_9446604Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zљtrace_0zњtrace_1

ћtrace_0
ќtrace_12б
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446614
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446624Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zћtrace_0zќtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_layer_call_fn_9446633Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_layer_call_and_return_conditional_losses_9446643Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
0
ќ0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_1_layer_call_fn_9445859lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_1_layer_call_fn_9446403inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_1_layer_call_fn_9446412inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
.__inference_sequential_1_layer_call_fn_9445927lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446428inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446444inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЃB 
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445937lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЃB 
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445947lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ё	variables
ђtrainable_variables
ѓregularization_losses
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
в
trace_0
trace_12
(__inference_lambda_layer_call_fn_9446648
(__inference_lambda_layer_call_fn_9446653Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Э
C__inference_lambda_layer_call_and_return_conditional_losses_9446660
C__inference_lambda_layer_call_and_return_conditional_losses_9446667Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_conv2d_layer_call_fn_9446676Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_conv2d_layer_call_and_return_conditional_losses_9446686Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_re_lu_layer_call_fn_9446691Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_re_lu_layer_call_and_return_conditional_losses_9446696Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
я
 trace_02а
)__inference_flatten_layer_call_fn_9446701Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0

Ёtrace_02ы
D__inference_flatten_layer_call_and_return_conditional_losses_9446707Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_layer_call_fn_9446002lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_layer_call_fn_9446453inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_layer_call_fn_9446462inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
,__inference_sequential_layer_call_fn_9446081lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446478inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446494inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446093lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446105lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
И
Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
ж
Їtrace_0
Јtrace_12
*__inference_lambda_1_layer_call_fn_9446712
*__inference_lambda_1_layer_call_fn_9446717Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЇtrace_0zЈtrace_1

Љtrace_0
Њtrace_12б
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446727
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446737Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЉtrace_0zЊtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
э
Аtrace_02Ю
'__inference_dense_layer_call_fn_9446746Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0

Бtrace_02щ
B__inference_dense_layer_call_and_return_conditional_losses_9446756Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0
 "
trackable_list_wrapper
0
Њ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_1_layer_call_fn_9446148lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_1_layer_call_fn_9446503inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_1_layer_call_fn_9446512inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
.__inference_sequential_1_layer_call_fn_9446216lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446528inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446544inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЃB 
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446226lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЃB 
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446236lambda_1_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
в
Зtrace_0
Иtrace_12
(__inference_lambda_layer_call_fn_9446761
(__inference_lambda_layer_call_fn_9446766Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЗtrace_0zИtrace_1

Йtrace_0
Кtrace_12Э
C__inference_lambda_layer_call_and_return_conditional_losses_9446773
C__inference_lambda_layer_call_and_return_conditional_losses_9446780Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЙtrace_0zКtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
ю
Рtrace_02Я
(__inference_conv2d_layer_call_fn_9446789Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zРtrace_0

Сtrace_02ъ
C__inference_conv2d_layer_call_and_return_conditional_losses_9446799Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zСtrace_0
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
э
Чtrace_02Ю
'__inference_re_lu_layer_call_fn_9446804Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЧtrace_0

Шtrace_02щ
B__inference_re_lu_layer_call_and_return_conditional_losses_9446809Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zШtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
я
Юtrace_02а
)__inference_flatten_layer_call_fn_9446814Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0

Яtrace_02ы
D__inference_flatten_layer_call_and_return_conditional_losses_9446820Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЯtrace_0
 "
trackable_list_wrapper
@
В0
Г1
Д2
Е3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_layer_call_fn_9446291lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_layer_call_fn_9446553inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_layer_call_fn_9446562inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
,__inference_sequential_layer_call_fn_9446370lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446578inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446594inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446382lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_9446394lambda_input"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
ќBљ
*__inference_lambda_1_layer_call_fn_9446599inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ќBљ
*__inference_lambda_1_layer_call_fn_9446604inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446614inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446624inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
лBи
'__inference_dense_layer_call_fn_9446633inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_layer_call_and_return_conditional_losses_9446643inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
њBї
(__inference_lambda_layer_call_fn_9446648inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њBї
(__inference_lambda_layer_call_fn_9446653inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_lambda_layer_call_and_return_conditional_losses_9446660inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_lambda_layer_call_and_return_conditional_losses_9446667inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
мBй
(__inference_conv2d_layer_call_fn_9446676inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_layer_call_and_return_conditional_losses_9446686inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
лBи
'__inference_re_lu_layer_call_fn_9446691inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_re_lu_layer_call_and_return_conditional_losses_9446696inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
нBк
)__inference_flatten_layer_call_fn_9446701inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_flatten_layer_call_and_return_conditional_losses_9446707inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
ќBљ
*__inference_lambda_1_layer_call_fn_9446712inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ќBљ
*__inference_lambda_1_layer_call_fn_9446717inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446727inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446737inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
лBи
'__inference_dense_layer_call_fn_9446746inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_layer_call_and_return_conditional_losses_9446756inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
њBї
(__inference_lambda_layer_call_fn_9446761inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њBї
(__inference_lambda_layer_call_fn_9446766inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_lambda_layer_call_and_return_conditional_losses_9446773inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_lambda_layer_call_and_return_conditional_losses_9446780inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
мBй
(__inference_conv2d_layer_call_fn_9446789inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_layer_call_and_return_conditional_losses_9446799inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
лBи
'__inference_re_lu_layer_call_fn_9446804inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_re_lu_layer_call_and_return_conditional_losses_9446809inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
нBк
)__inference_flatten_layer_call_fn_9446814inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_flatten_layer_call_and_return_conditional_losses_9446820inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 7
__inference_<lambda>_919Ђ

Ђ 
Њ " 	0
__inference_<lambda>_922Ђ

Ђ 
Њ "Њ ж
__inference_action_1944352ЗПЂЛ
ГЂЏ
ЁВ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ
observationЊ|
<
	direction/,
observation/directionџџџџџџџџџ
<
image30
observation/imageџџџџџџџџџ
Њ
~
actor_network_stategd
0-
actor_network_state/0џџџџџџџџџ
0-
actor_network_state/1џџџџџџџџџ

 
Њ "уВп

PolicyStep&
action
actionџџџџџџџџџ	
stateЊ

actor_network_statesp
63
state/actor_network_state/0џџџџџџџџџ
63
state/actor_network_state/1џџџџџџџџџ
infoЂ І
__inference_action_1944587Ђ
Ђџ
еВб
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџЄ
observationЊ
F
	direction96
time_step/observation/directionџџџџџџџџџ
F
image=:
time_step/observation/imageџџџџџџџџџ
 Њ

actor_network_state~
=:
"policy_state/actor_network_state/0џџџџџџџџџ
=:
"policy_state/actor_network_state/1џџџџџџџџџ

 
Њ "уВп

PolicyStep&
action
actionџџџџџџџџџ	
stateЊ

actor_network_statesp
63
state/actor_network_state/0џџџџџџџџџ
63
state/actor_network_state/1џџџџџџџџџ
infoЂ Г
C__inference_conv2d_layer_call_and_return_conditional_losses_9446686l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Г
C__inference_conv2d_layer_call_and_return_conditional_losses_9446799l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
(__inference_conv2d_layer_call_fn_9446676_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
(__inference_conv2d_layer_call_fn_9446789_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЂ
B__inference_dense_layer_call_and_return_conditional_losses_9446643\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ђ
B__inference_dense_layer_call_and_return_conditional_losses_9446756\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_dense_layer_call_fn_9446633O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџz
'__inference_dense_layer_call_fn_9446746O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЖ
#__inference_distribution_fn_1944797ЛЂЗ
ЏЂЋ
ЁВ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ
observationЊ|
<
	direction/,
observation/directionџџџџџџџџџ
<
image30
observation/imageџџџџџџџџџ
Њ
~
actor_network_stategd
0-
actor_network_state/0џџџџџџџџџ
0-
actor_network_state/1џџџџџџџџџ
Њ "ОВК

PolicyStep
actionѕёНЂЙ
`
BЊ?

atol 	

locџџџџџџџџџ	

rtol 	
JЊG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
Ђ
j
parameters
Ђ 
Ђ
jname+tfp.distributions.Deterministic_ACTTypeSpec 
stateЊ

actor_network_statesp
63
state/actor_network_state/0џџџџџџџџџ
63
state/actor_network_state/1џџџџџџџџџ
infoЂ Ј
D__inference_flatten_layer_call_and_return_conditional_losses_9446707`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџH
 Ј
D__inference_flatten_layer_call_and_return_conditional_losses_9446820`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџH
 
)__inference_flatten_layer_call_fn_9446701S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџH
)__inference_flatten_layer_call_fn_9446814S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџHе
%__inference_get_initial_state_1944813Ћ"Ђ
Ђ


batch_size 
Њ "Њ
~
actor_network_stategd
0-
actor_network_state/0џџџџџџџџџ
0-
actor_network_state/1џџџџџџџџџЉ
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446614`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ
 Љ
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446624`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ
 Љ
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446727`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ
 Љ
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446737`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_lambda_1_layer_call_fn_9446599S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ
*__inference_lambda_1_layer_call_fn_9446604S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџ
*__inference_lambda_1_layer_call_fn_9446712S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ
*__inference_lambda_1_layer_call_fn_9446717S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџЗ
C__inference_lambda_layer_call_and_return_conditional_losses_9446660p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 З
C__inference_lambda_layer_call_and_return_conditional_losses_9446667p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 З
C__inference_lambda_layer_call_and_return_conditional_losses_9446773p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 З
C__inference_lambda_layer_call_and_return_conditional_losses_9446780p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
(__inference_lambda_layer_call_fn_9446648c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
(__inference_lambda_layer_call_fn_9446653c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџ
(__inference_lambda_layer_call_fn_9446761c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
(__inference_lambda_layer_call_fn_9446766c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџЎ
B__inference_re_lu_layer_call_and_return_conditional_losses_9446696h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Ў
B__inference_re_lu_layer_call_and_return_conditional_losses_9446809h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
'__inference_re_lu_layer_call_fn_9446691[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
'__inference_re_lu_layer_call_fn_9446804[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЙ
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445937l?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445947l?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446226l?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446236l?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446428d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446444d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446528d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446544d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_sequential_1_layer_call_fn_9445859_?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9445927_?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446148_?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446216_?Ђ<
5Ђ2
(%
lambda_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446403W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446412W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446503W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_9446512W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџН
G__inference_sequential_layer_call_and_return_conditional_losses_9446093rEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 Н
G__inference_sequential_layer_call_and_return_conditional_losses_9446105rEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 Н
G__inference_sequential_layer_call_and_return_conditional_losses_9446382rEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 Н
G__inference_sequential_layer_call_and_return_conditional_losses_9446394rEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 З
G__inference_sequential_layer_call_and_return_conditional_losses_9446478l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 З
G__inference_sequential_layer_call_and_return_conditional_losses_9446494l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 З
G__inference_sequential_layer_call_and_return_conditional_losses_9446578l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 З
G__inference_sequential_layer_call_and_return_conditional_losses_9446594l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 
,__inference_sequential_layer_call_fn_9446002eEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446081eEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446291eEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446370eEЂB
;Ђ8
.+
lambda_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446453_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446462_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446553_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
,__inference_sequential_layer_call_fn_9446562_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџHё
%__inference_signature_wrapper_9445795ЧиЂд
Ђ 
ЬЊШ
.

0/discount 

0/discountџџџџџџџџџ
L
0/observation/direction1.
0/observation/directionџџџџџџџџџ
L
0/observation/image52
0/observation/imageџџџџџџџџџ
*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ
M
1/actor_network_state/02/
1/actor_network_state/0џџџџџџџџџ
M
1/actor_network_state/12/
1/actor_network_state/1џџџџџџџџџ"кЊж
&
action
actionџџџџџџџџџ	
U
state/actor_network_state/063
state/actor_network_state/0џџџџџџџџџ
U
state/actor_network_state/163
state/actor_network_state/1џџџџџџџџџљ
%__inference_signature_wrapper_9445804Я0Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "Њ
I
actor_network_state/00-
actor_network_state/0џџџџџџџџџ
I
actor_network_state/10-
actor_network_state/1џџџџџџџџџY
%__inference_signature_wrapper_94458120Ђ

Ђ 
Њ "Њ

int64
int64 	=
%__inference_signature_wrapper_9445816Ђ

Ђ 
Њ "Њ 