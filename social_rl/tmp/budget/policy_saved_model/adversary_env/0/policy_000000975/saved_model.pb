Ѕг 
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
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Уг
Ў
+adversary_env/ValueRnnNetwork/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+adversary_env/ValueRnnNetwork/dense_17/bias
Ї
?adversary_env/ValueRnnNetwork/dense_17/bias/Read/ReadVariableOpReadVariableOp+adversary_env/ValueRnnNetwork/dense_17/bias*
_output_shapes
:*
dtype0
Ж
-adversary_env/ValueRnnNetwork/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*>
shared_name/-adversary_env/ValueRnnNetwork/dense_17/kernel
Џ
Aadversary_env/ValueRnnNetwork/dense_17/kernel/Read/ReadVariableOpReadVariableOp-adversary_env/ValueRnnNetwork/dense_17/kernel*
_output_shapes

:(*
dtype0
п
Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
и
Wadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias/Read/ReadVariableOpReadVariableOpCadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias*
_output_shapes	
: *
dtype0
ћ
Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	( *`
shared_nameQOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
є
cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel/Read/ReadVariableOpReadVariableOpOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel*
_output_shapes
:	( *
dtype0
ч
Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *V
shared_nameGEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
р
Yadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel/Read/ReadVariableOpReadVariableOpEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel*
_output_shapes
:	  *
dtype0
ю
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
ч
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias*
_output_shapes
: *
dtype0
і
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
я
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel*
_output_shapes

:  *
dtype0
ю
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
ч
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias*
_output_shapes
: *
dtype0
ї
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ь *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
№
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel*
_output_shapes
:	Ь *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:
*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4
* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:4
*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
§
Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Љ*c
shared_nameTRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
і
fadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes	
:Љ*
dtype0

Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Љ*e
shared_nameVTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
џ
hadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel* 
_output_shapes
:
Љ*
dtype0

[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*l
shared_name][adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias

oadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias/Read/ReadVariableOpReadVariableOp[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias*
_output_shapes	
:*
dtype0
Ќ
gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*x
shared_nameigadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
Ѕ
{adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel/Read/ReadVariableOpReadVariableOpgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel* 
_output_shapes
:
*
dtype0

]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *n
shared_name_]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel

qadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel/Read/ReadVariableOpReadVariableOp]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel*
_output_shapes
:	 *
dtype0

cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias

wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias*
_output_shapes
: *
dtype0
І
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel

yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel*
_output_shapes

:  *
dtype0

cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias

wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias*
_output_shapes
: *
dtype0
Ї
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ь *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
 
yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel*
_output_shapes
:	Ь *
dtype0
v
dense_12/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_12/bias_1
o
#dense_12/bias_1/Read/ReadVariableOpReadVariableOpdense_12/bias_1*
_output_shapes
:
*
dtype0
~
dense_12/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:4
*"
shared_namedense_12/kernel_1
w
%dense_12/kernel_1/Read/ReadVariableOpReadVariableOpdense_12/kernel_1*
_output_shapes

:4
*
dtype0
v
conv2d_2/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/bias_1
o
#conv2d_2/bias_1/Read/ReadVariableOpReadVariableOpconv2d_2/bias_1*
_output_shapes
:*
dtype0

conv2d_2/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_2/kernel_1

%conv2d_2/kernel_1/Read/ReadVariableOpReadVariableOpconv2d_2/kernel_1*&
_output_shapes
:*
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
Лч
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ѕц
valueъцBцц Bоц
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
WQ
VARIABLE_VALUEconv2d_2/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d_2/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_12/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_12/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ЌЅ
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ЊЃ
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ЌЅ
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ЊЃ
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
Є
VARIABLE_VALUE]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ЎЇ
VARIABLE_VALUEgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ
VARIABLE_VALUE[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUETadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_2/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d_2/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_12/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_12/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUECadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE-adversary_env/ValueRnnNetwork/dense_17/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE+adversary_env/ValueRnnNetwork/dense_17/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
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

Т0
У1
Ф2*

Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses* 

Ы0
Ь1
Э2*
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
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
ы
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses
й_random_generator
к
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
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
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
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
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

х0
ц1
ч2*

ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses* 

ю0
я1
№2*
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
ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*
* 
* 
ы
і	variables
їtrainable_variables
јregularization_losses
љ	keras_api
њ__call__
+ћ&call_and_return_all_conditional_losses
ќ_random_generator
§
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
<
Т0
У1
Ф2
3
Ы4
Ь5
Э6*
* 
* 
* 
щ
ўlayer-0
џlayer_with_weights-0
џlayer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ђ
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Э
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses* 
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses* 
Ќ
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses

kernel
bias*
Ќ
Ј	variables
Љtrainable_variables
Њregularization_losses
Ћ	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses

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
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*
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
<
х0
ц1
ч2
Џ3
ю4
я5
№6*
* 
* 
* 
щ
Гlayer-0
Дlayer_with_weights-0
Дlayer-1
Еlayer-2
Жlayer-3
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses*
Ђ
Нlayer-0
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses* 
Э
Фlayer-0
Хlayer_with_weights-0
Хlayer-1
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses*
* 
* 
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 
* 
* 

б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses* 
Ќ
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses

 kernel
!bias*
Ќ
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses

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
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
і	variables
їtrainable_variables
јregularization_losses
њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses*
* 
* 
* 
* 

ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses* 
Я
ю	variables
яtrainable_variables
№regularization_losses
ё	keras_api
ђ__call__
+ѓ&call_and_return_all_conditional_losses

kernel
bias
!є_jit_compiled_convolution_op*

ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses* 

ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
 trace_3* 

Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses* 
Ќ
Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
Вtrace_0
Гtrace_1
Дtrace_2
Еtrace_3* 
:
Жtrace_0
Зtrace_1
Иtrace_2
Йtrace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses* 
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
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses*
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
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Ј	variables
Љtrainable_variables
Њregularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses* 
Я
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses

kernel
bias
!е_jit_compiled_convolution_op*

ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses* 

м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*
:
чtrace_0
шtrace_1
щtrace_2
ъtrace_3* 
:
ыtrace_0
ьtrace_1
эtrace_2
юtrace_3* 

я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses* 
* 
* 
* 

ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses* 
:
њtrace_0
ћtrace_1
ќtrace_2
§trace_3* 
:
ўtrace_0
џtrace_1
trace_2
trace_3* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ќ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses* 
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
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*
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
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses*
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
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

Џtrace_0
Аtrace_1* 

Бtrace_0
Вtrace_1* 

0
1*

0
1*
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
ю	variables
яtrainable_variables
№regularization_losses
ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses*

Иtrace_0* 

Йtrace_0* 
* 
* 
* 
* 

Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses* 

Пtrace_0* 

Рtrace_0* 
* 
* 
* 

Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Цtrace_0* 

Чtrace_0* 
* 
$
ў0
џ1
2
3*
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
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Эtrace_0
Юtrace_1* 

Яtrace_0
аtrace_1* 
* 


0* 
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
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses* 

жtrace_0
зtrace_1* 

иtrace_0
йtrace_1* 

0
1*

0
1*
* 

кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses*

пtrace_0* 

рtrace_0* 
* 

0
1*
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
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

цtrace_0
чtrace_1* 

шtrace_0
щtrace_1* 

0
1*

0
1*
* 

ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses*

яtrace_0* 

№trace_0* 
* 
* 
* 
* 

ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

іtrace_0* 

їtrace_0* 
* 
* 
* 

јnon_trainable_variables
љlayers
њmetrics
 ћlayer_regularization_losses
ќlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses* 

§trace_0* 

ўtrace_0* 
* 
$
Г0
Д1
Е2
Ж3*
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
џnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 


Н0* 
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

Ф0
Х1*
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

action_0_observation_imagePlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ

action_0_observation_random_zPlaceholder*'
_output_shapes
:џџџџџџџџџ2*
dtype0*
shape:џџџџџџџџџ2

action_0_observation_time_stepPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
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
ф
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_imageaction_0_observation_random_zaction_0_observation_time_stepaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1conv2d_2/kernel_1conv2d_2/bias_1dense_12/kernel_1dense_12/bias_1eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biaseadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26263211
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Н
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26263220
м
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26263232

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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26263228
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp%conv2d_2/kernel_1/Read/ReadVariableOp#conv2d_2/bias_1/Read/ReadVariableOp%dense_12/kernel_1/Read/ReadVariableOp#dense_12/bias_1/Read/ReadVariableOpyadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpwadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpyadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel/Read/ReadVariableOpwadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias/Read/ReadVariableOpqadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel/Read/ReadVariableOp{adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel/Read/ReadVariableOpoadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias/Read/ReadVariableOphadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpfadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpaadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel/Read/ReadVariableOp_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias/Read/ReadVariableOpaadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOp_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpYadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel/Read/ReadVariableOpcadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel/Read/ReadVariableOpWadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias/Read/ReadVariableOpAadversary_env/ValueRnnNetwork/dense_17/kernel/Read/ReadVariableOp?adversary_env/ValueRnnNetwork/dense_17/bias/Read/ReadVariableOpConst*(
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
GPU 2J 8 **
f%R#
!__inference__traced_save_26264549
 
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepconv2d_2/kernel_1conv2d_2/bias_1dense_12/kernel_1dense_12/bias_1eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biaseadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasconv2d_2/kernelconv2d_2/biasdense_12/kerneldense_12/biasMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernelKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/biasMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernelKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/biasEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernelOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernelCadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias-adversary_env/ValueRnnNetwork/dense_17/kernel+adversary_env/ValueRnnNetwork/dense_17/bias*'
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_26264640фе
П
F
*__inference_re_lu_2_layer_call_fn_26264349

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
к
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263280

inputs+
conv2d_2_26263259:
conv2d_2_26263261:
identityЂ conv2d_2/StatefulPartitionedCallТ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263246
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263259conv2d_2_26263261*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
в
-
+__inference_function_with_signature_1949163у
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
GPU 2J 8 *"
fR
__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 

V
&__inference_signature_wrapper_26263220

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
+__inference_function_with_signature_1949136a
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
Z

__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 

т
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263390
lambda_4_input+
conv2d_2_26263382:
conv2d_2_26263384:
identityЂ conv2d_2/StatefulPartitionedCallЪ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263324
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263382conv2d_2_26263384*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input

Ќ
/__inference_sequential_4_layer_call_fn_26263287
lambda_4_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263280p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
в
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263404

inputs
identityК
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263401i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_5_layer_call_fn_26264257

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263469`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264002

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
к
f
&__inference_signature_wrapper_26263228
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
+__inference_function_with_signature_1949152^
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
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263401

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

т
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263741
lambda_4_input+
conv2d_2_26263733:
conv2d_2_26263735:
identityЂ conv2d_2/StatefulPartitionedCallЪ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263675
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263733conv2d_2_26263735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
ј

J__inference_sequential_5_layer_call_and_return_conditional_losses_26264154

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ь
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
S
/__inference_sequential_6_layer_call_fn_26263442
lambda_6_input
identityН
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263434`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
ўЂ
С
__inference_action_1949408
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ь ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
Љn
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	Љ
identity	

identity_1

identity_2ЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
value	B :ђ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ђ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         н
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4М
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4А
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЬ
  р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ь *
dtype0ч
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ш
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
:ќ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
і
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
Љ*
dtype0Ќ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉѓ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:Љ*
dtype0З
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџЉ   Ё
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЉД
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
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RЈ
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџС
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2А
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
discount:b^
/
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:џџџџџџџџџ2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/1
С
S
/__inference_sequential_6_layer_call_fn_26263793
lambda_6_input
identityН
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263785`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
п
(
&__inference_signature_wrapper_26263232ѕ
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
+__inference_function_with_signature_1949163*(
_construction_contextkEagerRuntime*
_input_shapes 
Љ
K
/__inference_sequential_6_layer_call_fn_26264112

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263785`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263324

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
G
+__inference_lambda_4_layer_call_fn_26264180

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263324h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

в
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263488

inputs#
dense_12_26263482:4

dense_12_26263484:

identityЂ dense_12/StatefulPartitionedCallК
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263469
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263482dense_12_26263484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
H
,__inference_flatten_6_layer_call_fn_26264228

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_6_layer_call_fn_26264244

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263419`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
в
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263755

inputs
identityК
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263752i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
С
S
/__inference_sequential_6_layer_call_fn_26263407
lambda_6_input
identityН
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263404`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input

в
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263839

inputs#
dense_12_26263833:4

dense_12_26263835:

identityЂ dense_12/StatefulPartitionedCallК
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263820
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263833dense_12_26263835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ	
ї
F__inference_dense_12_layer_call_and_return_conditional_losses_26264301

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264252

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263523

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
 
__inference_action_1949035
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
time_step_5
policy_state
policy_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ь ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
Љn
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	Љ
identity	

identity_1

identity_2ЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
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
value	B :ь
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ш
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_4QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ш
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_5QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         н
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4М
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4А
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЬ
  р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ь *
dtype0ч
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ш
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
:ќ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
і
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
Љ*
dtype0Ќ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉѓ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:Љ*
dtype0З
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџЉ   Ё
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЉД
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
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RЈ
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџС
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2А
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
_user_specified_name	time_step:ZV
/
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ2
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state
щ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

/__inference_sequential_5_layer_call_fn_26264129

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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

Ї
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264102

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identityЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџj
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
K
/__inference_sequential_6_layer_call_fn_26263989

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263404`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264383

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Џ
к
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263583
lambda_5_input#
dense_12_26263577:4

dense_12_26263579:

identityЂ dense_12/StatefulPartitionedCallТ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263523
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263577dense_12_26263579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_5_input
љ
к
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263701

inputs+
conv2d_2_26263693:
conv2d_2_26263695:
identityЂ conv2d_2/StatefulPartitionedCallТ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263675
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263693conv2d_2_26263695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
`
__inference_<lambda>_3489!
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
ц
Є
/__inference_sequential_5_layer_call_fn_26263846
lambda_5_input
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
_user_specified_namelambda_5_input
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263419

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
С
G
+__inference_lambda_4_layer_call_fn_26264175

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263246h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С{
о
$__inference__traced_restore_26264640
file_prefix&
assignvariableop_global_step:	 >
$assignvariableop_1_conv2d_2_kernel_1:0
"assignvariableop_2_conv2d_2_bias_1:6
$assignvariableop_3_dense_12_kernel_1:4
0
"assignvariableop_4_dense_12_bias_1:

xassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel:	Ь 
vassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias: 
xassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel:  
vassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias: 
passignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel:	 
{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel:
~
oassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias:	|
hassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:
Љu
fassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:	Љ=
#assignvariableop_14_conv2d_2_kernel:/
!assignvariableop_15_conv2d_2_bias:5
#assignvariableop_16_dense_12_kernel:4
/
!assignvariableop_17_dense_12_bias:
t
aassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel:	Ь m
_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias: s
aassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel:  m
_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias: l
Yassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel:	  v
cassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel:	( f
Wassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias:	 S
Aassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernel:(M
?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_bias:
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
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_2_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dense_12_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ч
AssignVariableOp_5AssignVariableOpxassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_6AssignVariableOpvassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ч
AssignVariableOp_7AssignVariableOpxassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_8AssignVariableOpvassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_9AssignVariableOppassignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_10AssignVariableOp{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_11AssignVariableOpoassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_12AssignVariableOphassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_13AssignVariableOpfassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_12_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_12_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_18AssignVariableOpaassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_19AssignVariableOp_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_20AssignVariableOpaassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_21AssignVariableOp_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_22AssignVariableOpYassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_23AssignVariableOpcassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_24AssignVariableOpWassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_25AssignVariableOpAassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_26AssignVariableOp?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_biasIdentity_26:output:0"/device:CPU:0*
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
Љ
K
/__inference_sequential_6_layer_call_fn_26264107

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263755`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
њ
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264116

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264194

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264379

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_6_layer_call_fn_26264370

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263752`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
щ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

т
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263378
lambda_4_input+
conv2d_2_26263370:
conv2d_2_26263372:
identityЂ conv2d_2/StatefulPartitionedCallЪ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263246
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263370conv2d_2_26263372*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
ш
Є
/__inference_sequential_4_layer_call_fn_26263952

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263350p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ї
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263968

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identityЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџj
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

в
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263898

inputs#
dense_12_26263892:4

dense_12_26263894:

identityЂ dense_12/StatefulPartitionedCallК
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263874
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263892dense_12_26263894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц

+__inference_dense_12_layer_call_fn_26264422

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs

Ї
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264086

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identityЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџj
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263447
lambda_6_input
identityТ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263401i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
щ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264223

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_6_layer_call_fn_26264375

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263770`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264318

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264325

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
к
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263924
lambda_5_input#
dense_12_26263918:4

dense_12_26263920:

identityЂ dense_12/StatefulPartitionedCallТ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263820
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263918dense_12_26263920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_5_input
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264272

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
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
%__inference_get_initial_state_1949131

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
Ц

+__inference_dense_12_layer_call_fn_26264291

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs
ЇK
Ј
!__inference__traced_save_26264549
file_prefix*
&savev2_global_step_read_readvariableop	0
,savev2_conv2d_2_kernel_1_read_readvariableop.
*savev2_conv2d_2_bias_1_read_readvariableop0
,savev2_dense_12_kernel_1_read_readvariableop.
*savev2_dense_12_bias_1_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableop|
xsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopz
vsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias_read_readvariableops
osavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopq
msavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableopl
hsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel_read_readvariableopj
fsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias_read_readvariableopl
hsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel_read_readvariableopj
fsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias_read_readvariableopd
`savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel_read_readvariableopn
jsavev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel_read_readvariableopb
^savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias_read_readvariableopL
Hsavev2_adversary_env_valuernnnetwork_dense_17_kernel_read_readvariableopJ
Fsavev2_adversary_env_valuernnnetwork_dense_17_bias_read_readvariableop
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
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop,savev2_conv2d_2_kernel_1_read_readvariableop*savev2_conv2d_2_bias_1_read_readvariableop,savev2_dense_12_kernel_1_read_readvariableop*savev2_dense_12_bias_1_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableopxsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopvsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias_read_readvariableoposavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopmsavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias_read_readvariableop`savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel_read_readvariableopjsavev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel_read_readvariableop^savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias_read_readvariableopHsavev2_adversary_env_valuernnnetwork_dense_17_kernel_read_readvariableopFsavev2_adversary_env_valuernnnetwork_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*
_input_shapesљ
і: : :::4
:
:	Ь : :  : :	 :
::
Љ:Љ:::4
:
:	Ь : :  : :	  :	( : :(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:4
: 

_output_shapes
:
:%!

_output_shapes
:	Ь : 
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
::&"
 
_output_shapes
:
Љ:!

_output_shapes	
:Љ:,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:4
: 

_output_shapes
:
:%!

_output_shapes
:	Ь : 
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

Ќ
/__inference_sequential_4_layer_call_fn_26263638
lambda_4_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263631p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
ъ
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263452
lambda_6_input
identityТ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263419i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
Щ	
ї
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs
Њ

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264213

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_5_layer_call_fn_26264393

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263874`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263798
lambda_6_input
identityТ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263752i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
Ѕ

__inference_action_1949649
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_image"
time_step_observation_random_z#
time_step_observation_time_step&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ь ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
Љn
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	Љ
identity	

identity_1

identity_2ЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
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
value	B :ќ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ћ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_observation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :ь
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         н
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4М
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4А
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЬ
  р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ь *
dtype0ч
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ш
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
:ќ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
і
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
Љ*
dtype0Ќ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉѓ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:Љ*
dtype0З
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџЉ   Ё
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЉД
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
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RЈ
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџС
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2А
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
_user_specified_nametime_step/discount:lh
/
_output_shapes
:џџџџџџџџџ
5
_user_specified_nametime_step/observation/image:gc
'
_output_shapes
:џџџџџџџџџ2
8
_user_specified_name time_step/observation/random_z:hd
'
_output_shapes
:џџџџџџџџџ
9
_user_specified_name!time_step/observation/time_step:lh
(
_output_shapes
:џџџџџџџџџ
<
_user_specified_name$"policy_state/actor_network_state/0:lh
(
_output_shapes
:џџџџџџџџџ
<
_user_specified_name$"policy_state/actor_network_state/1
­

+__inference_function_with_signature_1949068
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1!
unknown:
	unknown_0:
	unknown_1:4

	unknown_2:

	unknown_3:	Ь 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:
Љ

unknown_11:	Љ
identity	

identity_1

identity_2ЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_imageobservation_random_zobservation_time_stepactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_action_1949035k
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
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 22
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
0/discount:d`
/
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:џџџџџџџџџ2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_name0/observation/time_step:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/1

U
%__inference_get_initial_state_1949881

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
С
S
/__inference_sequential_6_layer_call_fn_26263758
lambda_6_input
identityН
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263755`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
њ
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263998

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ј

J__inference_sequential_5_layer_call_and_return_conditional_losses_26264170

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ь
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263770

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264282

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264413

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
к
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263934
lambda_5_input#
dense_12_26263928:4

dense_12_26263930:

identityЂ dense_12/StatefulPartitionedCallТ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263874
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263928dense_12_26263930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_5_input
Ё
G
+__inference_lambda_5_layer_call_fn_26264388

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263820`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264120

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ю

/__inference_sequential_5_layer_call_fn_26264138

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
Њ

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264344

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
ъ
#__inference_distribution_fn_1949865
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ь ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
Љn
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	Љ
identity	

identity_1	

identity_2	

identity_3

identity_4ЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpЂgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
value	B :ђ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ђ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџц
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	М
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         н
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	Ж
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   л
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџГ
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4М
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4А
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЬ
  р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ь *
dtype0ч
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ш
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0щ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ђ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
:ќ
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
і
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
Љ*
dtype0Ќ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉѓ
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:Љ*
dtype0З
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЉ
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџЉ   Ё
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЉД
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
: А

Identity_3Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_4Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџС
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2в
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2А
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
discount:b^
/
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:џџџџџџџџџ2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameactor_network_state/1

Ї
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263984

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identityЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџj
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

/__inference_sequential_5_layer_call_fn_26264011

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263488o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263820

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш
Є
/__inference_sequential_4_layer_call_fn_26264070

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263701p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ќ
/__inference_sequential_4_layer_call_fn_26263366
lambda_4_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263350p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input

т
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263729
lambda_4_input+
conv2d_2_26263721:
conv2d_2_26263723:
identityЂ conv2d_2/StatefulPartitionedCallЪ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263597
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263721conv2d_2_26263723*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
в
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263785

inputs
identityК
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263770i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ц
Є
/__inference_sequential_5_layer_call_fn_26263495
lambda_5_input
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263488o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
_user_specified_namelambda_5_input
ш
Є
/__inference_sequential_4_layer_call_fn_26264061

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263631p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263246

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263675

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

в
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263547

inputs#
dense_12_26263541:4

dense_12_26263543:

identityЂ dense_12/StatefulPartitionedCallК
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263523
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263541dense_12_26263543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
G
+__inference_lambda_4_layer_call_fn_26264311

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263675h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263597

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

J__inference_sequential_5_layer_call_and_return_conditional_losses_26264036

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ь
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

/__inference_sequential_5_layer_call_fn_26264020

inputs
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263469

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

[
+__inference_function_with_signature_1949136

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
%__inference_get_initial_state_1949131a
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
Щ	
ї
F__inference_dense_12_layer_call_and_return_conditional_losses_26263832

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs
љ
к
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263631

inputs+
conv2d_2_26263610:
conv2d_2_26263612:
identityЂ conv2d_2/StatefulPartitionedCallТ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263597
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263610conv2d_2_26263612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263620з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
к
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263350

inputs+
conv2d_2_26263342:
conv2d_2_26263344:
identityЂ conv2d_2/StatefulPartitionedCallТ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263324
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_26263342conv2d_2_26263344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258у
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269з
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263277r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_6_layer_call_fn_26264239

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263401`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ю
 
+__inference_conv2d_2_layer_call_fn_26264334

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263609w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш
Є
/__inference_sequential_4_layer_call_fn_26263943

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263280p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю
 
+__inference_conv2d_2_layer_call_fn_26264203

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
H
,__inference_flatten_6_layer_call_fn_26264359

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_26263628a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264403

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
G
+__inference_lambda_5_layer_call_fn_26264262

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263523`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263874

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
value	B :4Ј
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263803
lambda_6_input
identityТ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263770i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:W S
'
_output_shapes
:џџџџџџџџџ2
(
_user_specified_namelambda_6_input
С
G
+__inference_lambda_4_layer_call_fn_26264306

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_26263597h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
Є
/__inference_sequential_5_layer_call_fn_26263563
lambda_5_input
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
_user_specified_namelambda_5_input
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264248

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Щ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264365

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
k
+__inference_function_with_signature_1949152
unknown:	 
identity	ЂStatefulPartitionedCall
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
GPU 2J 8 *"
fR
__inference_<lambda>_3489^
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

Ќ
/__inference_sequential_4_layer_call_fn_26263717
lambda_4_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263701p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_4_input
і
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263752

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ц
Є
/__inference_sequential_5_layer_call_fn_26263914
lambda_5_input
unknown:4

	unknown_0:

identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
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
_user_specified_namelambda_5_input
Й

&__inference_signature_wrapper_26263211
discount
observation_image
observation_random_z
observation_time_step

reward
	step_type
actor_network_state_0
actor_network_state_1!
unknown:
	unknown_0:
	unknown_1:4

	unknown_2:

	unknown_3:	Ь 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:
Љ

unknown_11:	Љ
identity	

identity_1

identity_2ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_imageobservation_random_zobservation_time_stepactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1949068k
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
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:d`
/
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:џџџџџџџџџ2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_name0/observation/time_step:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_name1/actor_network_state/1
в
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263434

inputs
identityК
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_26263419i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Щ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264234

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

J__inference_sequential_5_layer_call_and_return_conditional_losses_26264052

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ь
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџ4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264187

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ	
ї
F__inference_dense_12_layer_call_and_return_conditional_losses_26264432

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ4
 
_user_specified_nameinputs
щ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264354

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
K
/__inference_sequential_6_layer_call_fn_26263994

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263434`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
П
F
*__inference_re_lu_2_layer_call_fn_26264218

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26263269h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26263258

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
к
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263573
lambda_5_input#
dense_12_26263567:4

dense_12_26263569:

identityЂ dense_12/StatefulPartitionedCallТ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_26263469
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_26263567dense_12_26263569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_26263481x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_5_input"ПL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ч
actionМ
4

0/discount&
action_0_discount:0џџџџџџџџџ
R
0/observation/image;
action_0_observation_image:0џџџџџџџџџ
P
0/observation/random_z6
action_0_observation_random_z:0џџџџџџџџџ2
R
0/observation/time_step7
 action_0_observation_time_step:0џџџџџџџџџ
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
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:№б
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
__inference_action_1949408
__inference_action_1949649Ч
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
#__inference_distribution_fn_1949865Ћ
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
%__inference_get_initial_state_1949881І
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
ЏBЌ
__inference_<lambda>_3492"
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
ЏBЌ
__inference_<lambda>_3489"
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
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
x:v	Ь 2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias
w:u  2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias
p:n	 2]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel
{:y
2gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
j:h2[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias
h:f
Љ2Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
a:_Љ2Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
`:^	Ь 2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
_:]  2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
X:V	  2Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
b:`	( 2Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
R:P 2Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
?:=(2-adversary_env/ValueRnnNetwork/dense_17/kernel
9:72+adversary_env/ValueRnnNetwork/dense_17/bias
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
іBѓ
__inference_action_1949408	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"Ч
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
ЬBЩ
__inference_action_1949649time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/imagetime_step/observation/random_ztime_step/observation/time_step"policy_state/actor_network_state/0"policy_state/actor_network_state/1"Ч
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
уBр
#__inference_distribution_fn_1949865	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"Ћ
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
%__inference_get_initial_state_1949881
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
нBк
&__inference_signature_wrapper_26263211
0/discount0/observation/image0/observation/random_z0/observation/time_step0/reward0/step_type1/actor_network_state/01/actor_network_state/1"
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
аBЭ
&__inference_signature_wrapper_26263220
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
ТBП
&__inference_signature_wrapper_26263228"
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
ТBП
&__inference_signature_wrapper_26263232"
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
8
Т0
У1
Ф2"
trackable_list_wrapper
Ћ
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
8
Ы0
Ь1
Э2"
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
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
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
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses
й_random_generator
к
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
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
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
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
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
8
х0
ц1
ч2"
trackable_list_wrapper
Ћ
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
8
ю0
я1
№2"
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
ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
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
і	variables
їtrainable_variables
јregularization_losses
љ	keras_api
њ__call__
+ћ&call_and_return_all_conditional_losses
ќ_random_generator
§
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
X
Т0
У1
Ф2
3
Ы4
Ь5
Э6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

ўlayer-0
џlayer_with_weights-0
џlayer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
О
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
ч
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
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
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
С
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
С
Ј	variables
Љtrainable_variables
Њregularization_losses
Ћ	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses

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
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
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
X
х0
ц1
ч2
Џ3
ю4
я5
№6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

Гlayer-0
Дlayer_with_weights-0
Дlayer-1
Еlayer-2
Жlayer-3
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_sequential
О
Нlayer-0
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_sequential
ч
Фlayer-0
Хlayer_with_weights-0
Хlayer-1
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
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
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"
_tf_keras_layer
С
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
С
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses

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
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
і	variables
їtrainable_variables
јregularization_losses
њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
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
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
ю	variables
яtrainable_variables
№regularization_losses
ё	keras_api
ђ__call__
+ѓ&call_and_return_all_conditional_losses

kernel
bias
!є_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses"
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
њ
trace_0
trace_1
trace_2
trace_32
/__inference_sequential_4_layer_call_fn_26263287
/__inference_sequential_4_layer_call_fn_26263943
/__inference_sequential_4_layer_call_fn_26263952
/__inference_sequential_4_layer_call_fn_26263366Р
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
 ztrace_0ztrace_1ztrace_2ztrace_3
ц
trace_0
trace_1
trace_2
trace_32ѓ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263968
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263984
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263378
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263390Р
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
 ztrace_0ztrace_1ztrace_2ztrace_3
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
њ
trace_0
trace_1
trace_2
trace_32
/__inference_sequential_6_layer_call_fn_26263407
/__inference_sequential_6_layer_call_fn_26263989
/__inference_sequential_6_layer_call_fn_26263994
/__inference_sequential_6_layer_call_fn_26263442Р
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
 ztrace_0ztrace_1ztrace_2ztrace_3
ц
trace_0
trace_1
trace_2
 trace_32ѓ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263998
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264002
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263447
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263452Р
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
 ztrace_0ztrace_1ztrace_2z trace_3
Ћ
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
С
Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses

kernel
bias"
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
­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
њ
Вtrace_0
Гtrace_1
Дtrace_2
Еtrace_32
/__inference_sequential_5_layer_call_fn_26263495
/__inference_sequential_5_layer_call_fn_26264011
/__inference_sequential_5_layer_call_fn_26264020
/__inference_sequential_5_layer_call_fn_26263563Р
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
 zВtrace_0zГtrace_1zДtrace_2zЕtrace_3
ц
Жtrace_0
Зtrace_1
Иtrace_2
Йtrace_32ѓ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264036
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264052
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263573
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263583Р
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
 zЖtrace_0zЗtrace_1zИtrace_2zЙtrace_3
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
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
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
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
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
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Ј	variables
Љtrainable_variables
Њregularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
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
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses

kernel
bias
!е_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses"
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
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
њ
чtrace_0
шtrace_1
щtrace_2
ъtrace_32
/__inference_sequential_4_layer_call_fn_26263638
/__inference_sequential_4_layer_call_fn_26264061
/__inference_sequential_4_layer_call_fn_26264070
/__inference_sequential_4_layer_call_fn_26263717Р
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
 zчtrace_0zшtrace_1zщtrace_2zъtrace_3
ц
ыtrace_0
ьtrace_1
эtrace_2
юtrace_32ѓ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264086
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264102
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263729
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263741Р
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
 zыtrace_0zьtrace_1zэtrace_2zюtrace_3
Ћ
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
њ
њtrace_0
ћtrace_1
ќtrace_2
§trace_32
/__inference_sequential_6_layer_call_fn_26263758
/__inference_sequential_6_layer_call_fn_26264107
/__inference_sequential_6_layer_call_fn_26264112
/__inference_sequential_6_layer_call_fn_26263793Р
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
 zњtrace_0zћtrace_1zќtrace_2z§trace_3
ц
ўtrace_0
џtrace_1
trace_2
trace_32ѓ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264116
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264120
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263798
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263803Р
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
 zўtrace_0zџtrace_1ztrace_2ztrace_3
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
С
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
њ
trace_0
trace_1
trace_2
trace_32
/__inference_sequential_5_layer_call_fn_26263846
/__inference_sequential_5_layer_call_fn_26264129
/__inference_sequential_5_layer_call_fn_26264138
/__inference_sequential_5_layer_call_fn_26263914Р
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
 ztrace_0ztrace_1ztrace_2ztrace_3
ц
trace_0
trace_1
trace_2
trace_32ѓ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264154
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264170
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263924
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263934Р
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
 ztrace_0ztrace_1ztrace_2ztrace_3
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
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
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
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
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
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
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
и
Џtrace_0
Аtrace_12
+__inference_lambda_4_layer_call_fn_26264175
+__inference_lambda_4_layer_call_fn_26264180Р
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
 zЏtrace_0zАtrace_1

Бtrace_0
Вtrace_12г
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264187
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264194Р
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
 zБtrace_0zВtrace_1
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
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
ю	variables
яtrainable_variables
№regularization_losses
ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
ё
Иtrace_02в
+__inference_conv2d_2_layer_call_fn_26264203Ђ
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
 zИtrace_0

Йtrace_02э
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264213Ђ
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
 zЙtrace_0
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
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
№
Пtrace_02б
*__inference_re_lu_2_layer_call_fn_26264218Ђ
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
 zПtrace_0

Рtrace_02ь
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264223Ђ
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ђ
Цtrace_02г
,__inference_flatten_6_layer_call_fn_26264228Ђ
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
 zЦtrace_0

Чtrace_02ю
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264234Ђ
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
 "
trackable_list_wrapper
@
ў0
џ1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_4_layer_call_fn_26263287lambda_4_input"Р
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
Bў
/__inference_sequential_4_layer_call_fn_26263943inputs"Р
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
Bў
/__inference_sequential_4_layer_call_fn_26263952inputs"Р
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
B
/__inference_sequential_4_layer_call_fn_26263366lambda_4_input"Р
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
B
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263968inputs"Р
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
B
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263984inputs"Р
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
ЄBЁ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263378lambda_4_input"Р
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
ЄBЁ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263390lambda_4_input"Р
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
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
и
Эtrace_0
Юtrace_12
+__inference_lambda_6_layer_call_fn_26264239
+__inference_lambda_6_layer_call_fn_26264244Р
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
 zЭtrace_0zЮtrace_1

Яtrace_0
аtrace_12г
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264248
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264252Р
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
 zЯtrace_0zаtrace_1
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_6_layer_call_fn_26263407lambda_6_input"Р
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
Bў
/__inference_sequential_6_layer_call_fn_26263989inputs"Р
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
Bў
/__inference_sequential_6_layer_call_fn_26263994inputs"Р
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
B
/__inference_sequential_6_layer_call_fn_26263442lambda_6_input"Р
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
B
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263998inputs"Р
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
B
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264002inputs"Р
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
ЄBЁ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263447lambda_6_input"Р
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
ЄBЁ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263452lambda_6_input"Р
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
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
и
жtrace_0
зtrace_12
+__inference_lambda_5_layer_call_fn_26264257
+__inference_lambda_5_layer_call_fn_26264262Р
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
 zжtrace_0zзtrace_1

иtrace_0
йtrace_12г
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264272
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264282Р
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
 zиtrace_0zйtrace_1
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
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
ё
пtrace_02в
+__inference_dense_12_layer_call_fn_26264291Ђ
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
 zпtrace_0

рtrace_02э
F__inference_dense_12_layer_call_and_return_conditional_losses_26264301Ђ
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
 zрtrace_0
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_5_layer_call_fn_26263495lambda_5_input"Р
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
Bў
/__inference_sequential_5_layer_call_fn_26264011inputs"Р
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
Bў
/__inference_sequential_5_layer_call_fn_26264020inputs"Р
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
B
/__inference_sequential_5_layer_call_fn_26263563lambda_5_input"Р
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264036inputs"Р
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264052inputs"Р
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
ЄBЁ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263573lambda_5_input"Р
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
ЄBЁ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263583lambda_5_input"Р
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
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
и
цtrace_0
чtrace_12
+__inference_lambda_4_layer_call_fn_26264306
+__inference_lambda_4_layer_call_fn_26264311Р
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
 zцtrace_0zчtrace_1

шtrace_0
щtrace_12г
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264318
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264325Р
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
 zшtrace_0zщtrace_1
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
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
ё
яtrace_02в
+__inference_conv2d_2_layer_call_fn_26264334Ђ
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
 zяtrace_0

№trace_02э
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264344Ђ
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
 z№trace_0
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
ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
№
іtrace_02б
*__inference_re_lu_2_layer_call_fn_26264349Ђ
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
 zіtrace_0

їtrace_02ь
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264354Ђ
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
 zїtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
јnon_trainable_variables
љlayers
њmetrics
 ћlayer_regularization_losses
ќlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
ђ
§trace_02г
,__inference_flatten_6_layer_call_fn_26264359Ђ
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
 z§trace_0

ўtrace_02ю
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264365Ђ
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
 zўtrace_0
 "
trackable_list_wrapper
@
Г0
Д1
Е2
Ж3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_4_layer_call_fn_26263638lambda_4_input"Р
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
Bў
/__inference_sequential_4_layer_call_fn_26264061inputs"Р
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
Bў
/__inference_sequential_4_layer_call_fn_26264070inputs"Р
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
B
/__inference_sequential_4_layer_call_fn_26263717lambda_4_input"Р
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
B
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264086inputs"Р
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
B
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264102inputs"Р
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
ЄBЁ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263729lambda_4_input"Р
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
ЄBЁ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263741lambda_4_input"Р
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
џnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
и
trace_0
trace_12
+__inference_lambda_6_layer_call_fn_26264370
+__inference_lambda_6_layer_call_fn_26264375Р
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
 ztrace_0ztrace_1

trace_0
trace_12г
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264379
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264383Р
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
 ztrace_0ztrace_1
 "
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_6_layer_call_fn_26263758lambda_6_input"Р
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
Bў
/__inference_sequential_6_layer_call_fn_26264107inputs"Р
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
Bў
/__inference_sequential_6_layer_call_fn_26264112inputs"Р
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
B
/__inference_sequential_6_layer_call_fn_26263793lambda_6_input"Р
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
B
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264116inputs"Р
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
B
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264120inputs"Р
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
ЄBЁ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263798lambda_6_input"Р
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
ЄBЁ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263803lambda_6_input"Р
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
и
trace_0
trace_12
+__inference_lambda_5_layer_call_fn_26264388
+__inference_lambda_5_layer_call_fn_26264393Р
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
 ztrace_0ztrace_1

trace_0
trace_12г
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264403
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264413Р
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
 ztrace_0ztrace_1
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_dense_12_layer_call_fn_26264422Ђ
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
 ztrace_0

trace_02э
F__inference_dense_12_layer_call_and_return_conditional_losses_26264432Ђ
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
 ztrace_0
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_5_layer_call_fn_26263846lambda_5_input"Р
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
Bў
/__inference_sequential_5_layer_call_fn_26264129inputs"Р
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
Bў
/__inference_sequential_5_layer_call_fn_26264138inputs"Р
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
B
/__inference_sequential_5_layer_call_fn_26263914lambda_5_input"Р
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264154inputs"Р
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264170inputs"Р
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
ЄBЁ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263924lambda_5_input"Р
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
ЄBЁ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263934lambda_5_input"Р
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
§Bњ
+__inference_lambda_4_layer_call_fn_26264175inputs"Р
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
§Bњ
+__inference_lambda_4_layer_call_fn_26264180inputs"Р
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
B
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264187inputs"Р
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
B
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264194inputs"Р
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
пBм
+__inference_conv2d_2_layer_call_fn_26264203inputs"Ђ
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
њBї
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264213inputs"Ђ
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
оBл
*__inference_re_lu_2_layer_call_fn_26264218inputs"Ђ
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
љBі
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264223inputs"Ђ
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
рBн
,__inference_flatten_6_layer_call_fn_26264228inputs"Ђ
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
ћBј
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264234inputs"Ђ
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
§Bњ
+__inference_lambda_6_layer_call_fn_26264239inputs"Р
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
§Bњ
+__inference_lambda_6_layer_call_fn_26264244inputs"Р
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
B
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264248inputs"Р
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
B
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264252inputs"Р
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
§Bњ
+__inference_lambda_5_layer_call_fn_26264257inputs"Р
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
§Bњ
+__inference_lambda_5_layer_call_fn_26264262inputs"Р
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
B
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264272inputs"Р
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
B
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264282inputs"Р
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
пBм
+__inference_dense_12_layer_call_fn_26264291inputs"Ђ
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
њBї
F__inference_dense_12_layer_call_and_return_conditional_losses_26264301inputs"Ђ
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
§Bњ
+__inference_lambda_4_layer_call_fn_26264306inputs"Р
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
§Bњ
+__inference_lambda_4_layer_call_fn_26264311inputs"Р
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
B
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264318inputs"Р
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
B
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264325inputs"Р
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
пBм
+__inference_conv2d_2_layer_call_fn_26264334inputs"Ђ
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
њBї
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264344inputs"Ђ
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
оBл
*__inference_re_lu_2_layer_call_fn_26264349inputs"Ђ
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
љBі
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264354inputs"Ђ
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
рBн
,__inference_flatten_6_layer_call_fn_26264359inputs"Ђ
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
ћBј
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264365inputs"Ђ
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
§Bњ
+__inference_lambda_6_layer_call_fn_26264370inputs"Р
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
§Bњ
+__inference_lambda_6_layer_call_fn_26264375inputs"Р
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
B
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264379inputs"Р
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
B
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264383inputs"Р
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
§Bњ
+__inference_lambda_5_layer_call_fn_26264388inputs"Р
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
§Bњ
+__inference_lambda_5_layer_call_fn_26264393inputs"Р
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
B
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264403inputs"Р
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
B
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264413inputs"Р
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
пBм
+__inference_dense_12_layer_call_fn_26264422inputs"Ђ
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
њBї
F__inference_dense_12_layer_call_and_return_conditional_losses_26264432inputs"Ђ
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
 8
__inference_<lambda>_3489Ђ

Ђ 
Њ " 	1
__inference_<lambda>_3492Ђ

Ђ 
Њ "Њ 
__inference_action_1949408ѕ§Ђљ
ёЂэ
пВл
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџЬ
observationМЊИ
<
image30
observation/imageџџџџџџџџџ
:
random_z.+
observation/random_zџџџџџџџџџ2
<
	time_step/,
observation/time_stepџџџџџџџџџ
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
infoЂ ь
__inference_action_1949649ЭеЂб
ЩЂХ
В
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџъ
observationкЊж
F
image=:
time_step/observation/imageџџџџџџџџџ
D
random_z85
time_step/observation/random_zџџџџџџџџџ2
F
	time_step96
time_step/observation/time_stepџџџџџџџџџ
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
infoЂ Ж
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264213l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Ж
F__inference_conv2d_2_layer_call_and_return_conditional_losses_26264344l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_2_layer_call_fn_26264203_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
+__inference_conv2d_2_layer_call_fn_26264334_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџІ
F__inference_dense_12_layer_call_and_return_conditional_losses_26264301\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ4
Њ "%Ђ"

0џџџџџџџџџ

 І
F__inference_dense_12_layer_call_and_return_conditional_losses_26264432\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ4
Њ "%Ђ"

0џџџџџџџџџ

 ~
+__inference_dense_12_layer_call_fn_26264291O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ4
Њ "џџџџџџџџџ
~
+__inference_dense_12_layer_call_fn_26264422O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ4
Њ "џџџџџџџџџ
є
#__inference_distribution_fn_1949865ЬљЂѕ
эЂщ
пВл
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџЬ
observationМЊИ
<
image30
observation/imageџџџџџџџџџ
:
random_z.+
observation/random_zџџџџџџџџџ2
<
	time_step/,
observation/time_stepџџџџџџџџџ
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
infoЂ Ќ
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264234a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Ќ
G__inference_flatten_6_layer_call_and_return_conditional_losses_26264365a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_flatten_6_layer_call_fn_26264228T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
,__inference_flatten_6_layer_call_fn_26264359T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџе
%__inference_get_initial_state_1949881Ћ"Ђ
Ђ


batch_size 
Њ "Њ
~
actor_network_stategd
0-
actor_network_state/0џџџџџџџџџ
0-
actor_network_state/1џџџџџџџџџК
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264187p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264194p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264318p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_4_layer_call_and_return_conditional_losses_26264325p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_lambda_4_layer_call_fn_26264175c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
+__inference_lambda_4_layer_call_fn_26264180c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџ
+__inference_lambda_4_layer_call_fn_26264306c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
+__inference_lambda_4_layer_call_fn_26264311c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџЊ
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264272`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ4
 Њ
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264282`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ4
 Њ
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264403`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ4
 Њ
F__inference_lambda_5_layer_call_and_return_conditional_losses_26264413`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ4
 
+__inference_lambda_5_layer_call_fn_26264257S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ4
+__inference_lambda_5_layer_call_fn_26264262S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџ4
+__inference_lambda_5_layer_call_fn_26264388S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ4
+__inference_lambda_5_layer_call_fn_26264393S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџ4Њ
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264248`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p 
Њ "%Ђ"

0џџџџџџџџџ2
 Њ
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264252`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p
Њ "%Ђ"

0џџџџџџџџџ2
 Њ
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264379`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p 
Њ "%Ђ"

0џџџџџџџџџ2
 Њ
F__inference_lambda_6_layer_call_and_return_conditional_losses_26264383`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p
Њ "%Ђ"

0џџџџџџџџџ2
 
+__inference_lambda_6_layer_call_fn_26264239S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p 
Њ "џџџџџџџџџ2
+__inference_lambda_6_layer_call_fn_26264244S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p
Њ "џџџџџџџџџ2
+__inference_lambda_6_layer_call_fn_26264370S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p 
Њ "џџџџџџџџџ2
+__inference_lambda_6_layer_call_fn_26264375S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2

 
p
Њ "џџџџџџџџџ2Б
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264223h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Б
E__inference_re_lu_2_layer_call_and_return_conditional_losses_26264354h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
*__inference_re_lu_2_layer_call_fn_26264218[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
*__inference_re_lu_2_layer_call_fn_26264349[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџУ
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263378uGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 У
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263390uGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 У
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263729uGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 У
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263741uGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Л
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263968m?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Л
J__inference_sequential_4_layer_call_and_return_conditional_losses_26263984m?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Л
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264086m?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Л
J__inference_sequential_4_layer_call_and_return_conditional_losses_26264102m?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 
/__inference_sequential_4_layer_call_fn_26263287hGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26263366hGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26263638hGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26263717hGЂD
=Ђ:
0-
lambda_4_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26263943`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26263952`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26264061`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_4_layer_call_fn_26264070`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџК
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263573l?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 К
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263583l?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 К
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263924l?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 К
J__inference_sequential_5_layer_call_and_return_conditional_losses_26263934l?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 В
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264036d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 В
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264052d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 В
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264154d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 В
J__inference_sequential_5_layer_call_and_return_conditional_losses_26264170d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
/__inference_sequential_5_layer_call_fn_26263495_?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26263563_?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26263846_?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26263914_?Ђ<
5Ђ2
(%
lambda_5_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26264011W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26264020W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26264129W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

/__inference_sequential_5_layer_call_fn_26264138W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
Ж
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263447h?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ж
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263452h?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ж
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263798h?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ж
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263803h?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ў
J__inference_sequential_6_layer_call_and_return_conditional_losses_26263998`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ў
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264002`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ў
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264116`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ў
J__inference_sequential_6_layer_call_and_return_conditional_losses_26264120`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 
/__inference_sequential_6_layer_call_fn_26263407[?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p 

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26263442[?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26263758[?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p 

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26263793[?Ђ<
5Ђ2
(%
lambda_6_inputџџџџџџџџџ2
p

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26263989S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p 

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26263994S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26264107S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p 

 
Њ "џџџџџџџџџ2
/__inference_sequential_6_layer_call_fn_26264112S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ2
p

 
Њ "џџџџџџџџџ2О
&__inference_signature_wrapper_26263211ЄЂ 
Ђ 
Њ
.

0/discount 

0/discountџџџџџџџџџ
L
0/observation/image52
0/observation/imageџџџџџџџџџ
J
0/observation/random_z0-
0/observation/random_zџџџџџџџџџ2
L
0/observation/time_step1.
0/observation/time_stepџџџџџџџџџ
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
state/actor_network_state/1џџџџџџџџџњ
&__inference_signature_wrapper_26263220Я0Ђ-
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
actor_network_state/1џџџџџџџџџZ
&__inference_signature_wrapper_262632280Ђ

Ђ 
Њ "Њ

int64
int64 	>
&__inference_signature_wrapper_26263232Ђ

Ђ 
Њ "Њ 