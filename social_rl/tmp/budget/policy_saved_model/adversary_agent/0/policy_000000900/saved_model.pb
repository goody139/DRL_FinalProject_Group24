ёё
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
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018зќ
В
-adversary_agent/ValueRnnNetwork/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-adversary_agent/ValueRnnNetwork/dense_11/bias
Ћ
Aadversary_agent/ValueRnnNetwork/dense_11/bias/Read/ReadVariableOpReadVariableOp-adversary_agent/ValueRnnNetwork/dense_11/bias*
_output_shapes
:*
dtype0
К
/adversary_agent/ValueRnnNetwork/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*@
shared_name1/adversary_agent/ValueRnnNetwork/dense_11/kernel
Г
Cadversary_agent/ValueRnnNetwork/dense_11/kernel/Read/ReadVariableOpReadVariableOp/adversary_agent/ValueRnnNetwork/dense_11/kernel*
_output_shapes

:(*
dtype0
у
Eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *V
shared_nameGEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias
м
Yadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/Read/ReadVariableOpReadVariableOpEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias*
_output_shapes	
: *
dtype0
џ
Qadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	( *b
shared_nameSQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel
ј
eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel/Read/ReadVariableOpReadVariableOpQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel*
_output_shapes
:	( *
dtype0
ы
Gadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *X
shared_nameIGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel
ф
[adversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel/Read/ReadVariableOpReadVariableOpGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel*
_output_shapes
:	  *
dtype0
ђ
Madversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias
ы
aadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpReadVariableOpMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias*
_output_shapes
: *
dtype0
њ
Oadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *`
shared_nameQOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel
ѓ
cadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpReadVariableOpOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel*
_output_shapes

:  *
dtype0
№
Ladversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *]
shared_nameNLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias
щ
`adversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias/Read/ReadVariableOpReadVariableOpLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias*
_output_shapes
: *
dtype0
ј
Nadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *_
shared_namePNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel
ё
badversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel/Read/ReadVariableOpReadVariableOpNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel*
_output_shapes

:M *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0

Tadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
љ
hadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes
:*
dtype0

Vadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*g
shared_nameXVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel

jadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel*
_output_shapes
:	*
dtype0

]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*n
shared_name_]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias

qadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias/Read/ReadVariableOpReadVariableOp]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias*
_output_shapes	
:*
dtype0
А
iadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*z
shared_namekiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel
Љ
}adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel* 
_output_shapes
:
*
dtype0

_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *p
shared_namea_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel

sadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel/Read/ReadVariableOpReadVariableOp_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel*
_output_shapes
:	 *
dtype0
 
dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *u
shared_namefdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias

xadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias/Read/ReadVariableOpReadVariableOpdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias*
_output_shapes
: *
dtype0
Ј
fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *w
shared_namehfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel
Ё
zadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel/Read/ReadVariableOpReadVariableOpfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel*
_output_shapes

:  *
dtype0
 
dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *u
shared_namefdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias

xadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOpReadVariableOpdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias*
_output_shapes
: *
dtype0
Ј
fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *w
shared_namehfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel
Ё
zadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpReadVariableOpfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel*
_output_shapes

:M *
dtype0
v
conv2d_1/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/bias_1
o
#conv2d_1/bias_1/Read/ReadVariableOpReadVariableOpconv2d_1/bias_1*
_output_shapes
:*
dtype0

conv2d_1/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_1/kernel_1

%conv2d_1/kernel_1/Read/ReadVariableOpReadVariableOpconv2d_1/kernel_1*&
_output_shapes
:*
dtype0
t
dense_6/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias_1
m
"dense_6/bias_1/Read/ReadVariableOpReadVariableOpdense_6/bias_1*
_output_shapes
:*
dtype0
|
dense_6/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_6/kernel_1
u
$dense_6/kernel_1/Read/ReadVariableOpReadVariableOpdense_6/kernel_1*
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
сд
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*д
valueдBд Bд
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
VP
VARIABLE_VALUEdense_6/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_6/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_1/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d_1/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
­І
VARIABLE_VALUEfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ЋЄ
VARIABLE_VALUEdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
­І
VARIABLE_VALUEfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ЋЄ
VARIABLE_VALUEdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
І
VARIABLE_VALUE_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
АЉ
VARIABLE_VALUEiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
Ѕ
VARIABLE_VALUE]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUETadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_6/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_6/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_1/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d_1/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE/adversary_agent/ValueRnnNetwork/dense_11/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE-adversary_agent/ValueRnnNetwork/dense_11/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
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
а
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_directionaction_0_observation_imageaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1dense_6/kernel_1dense_6/bias_1conv2d_1/kernel_1conv2d_1/bias_1fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kerneliadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_24445028
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
&__inference_signature_wrapper_24445037
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
&__inference_signature_wrapper_24445049
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
&__inference_signature_wrapper_24445045
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp$dense_6/kernel_1/Read/ReadVariableOp"dense_6/bias_1/Read/ReadVariableOp%conv2d_1/kernel_1/Read/ReadVariableOp#conv2d_1/bias_1/Read/ReadVariableOpzadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpxadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOpzadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel/Read/ReadVariableOpxadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias/Read/ReadVariableOpsadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel/Read/ReadVariableOp}adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel/Read/ReadVariableOpqadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias/Read/ReadVariableOpjadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOphadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOpbadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel/Read/ReadVariableOp`adversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias/Read/ReadVariableOpcadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpaadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOp[adversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel/Read/ReadVariableOpeadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel/Read/ReadVariableOpYadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/Read/ReadVariableOpCadversary_agent/ValueRnnNetwork/dense_11/kernel/Read/ReadVariableOpAadversary_agent/ValueRnnNetwork/dense_11/bias/Read/ReadVariableOpConst*(
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
!__inference__traced_save_24446169
К
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepdense_6/kernel_1dense_6/bias_1conv2d_1/kernel_1conv2d_1/bias_1fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kerneliadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasdense_6/kerneldense_6/biasconv2d_1/kernelconv2d_1/biasNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernelLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/biasOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernelMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/biasGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernelQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernelEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/adversary_agent/ValueRnnNetwork/dense_11/kernel-adversary_agent/ValueRnnNetwork/dense_11/bias*'
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
$__inference__traced_restore_24446260е
Є
з
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445170
lambda_3_input"
dense_6_24445164:
dense_6_24445166:
identityЂdense_6/StatefulPartitionedCallТ
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445066
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445164dense_6_24445166*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_3_input

т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445627
lambda_2_input+
conv2d_1_24445619:
conv2d_1_24445621:
identityЂ conv2d_1/StatefulPartitionedCallЪ
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445561
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445619conv2d_1_24445621*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
щ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217

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
Ш	
і
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367

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
ц
Є
/__inference_sequential_3_layer_call_fn_24445092
lambda_3_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445085o
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
_user_specified_namelambda_3_input

ђ
&__inference_signature_wrapper_24445028
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
+__inference_function_with_signature_1946529k
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

т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445338
lambda_2_input+
conv2d_1_24445330:
conv2d_1_24445332:
identityЂ conv2d_1/StatefulPartitionedCallЪ
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445272
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445330conv2d_1_24445332*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
Ё
G
+__inference_lambda_3_layer_call_fn_24445832

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445066`
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
Ш	
і
E__inference_dense_6_layer_call_and_return_conditional_losses_24445989

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
Ф

*__inference_dense_6_layer_call_fn_24445979

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallк
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367o
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
Є
з
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445180
lambda_3_input"
dense_6_24445174:
dense_6_24445176:
identityЂdense_6/StatefulPartitionedCallТ
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445120
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445174dense_6_24445176*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_3_input
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445847

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
ф

J__inference_sequential_3_layer_call_and_return_conditional_losses_24445677

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф

J__inference_sequential_3_layer_call_and_return_conditional_losses_24445761

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­
ф
__inference_action_1947097
time_step_step_type
time_step_reward
time_step_discount#
time_step_observation_direction
time_step_observation_image&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpЂrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЃ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   п
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЛ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2ш
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2А
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
Е

__inference_action_1946862
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpЂrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЃ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   п
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЛ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2ш
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2А
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
ц
Є
/__inference_sequential_2_layer_call_fn_24445686

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445228o
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
Ё
G
+__inference_lambda_3_layer_call_fn_24445837

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445120`
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
ю
 
+__inference_conv2d_1_layer_call_fn_24445909

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallу
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206w
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
С
G
+__inference_lambda_2_layer_call_fn_24445886

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445272h
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
ѓ
`
__inference_<lambda>_2176!
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
ћ
С
#__inference_distribution_fn_1947307
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1	

identity_2	

identity_3

identity_4ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpЂrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЃ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   п
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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
: А

Identity_3Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_4Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЛ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
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
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2ш
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2А
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

Ї
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445811

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identityЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф

J__inference_sequential_3_layer_call_and_return_conditional_losses_24445661

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ

џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445919

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
Z

__inference_<lambda>_2179*(
_construction_contextkEagerRuntime*
_input_shapes 
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445900

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
ц
Є
/__inference_sequential_2_layer_call_fn_24445695

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445298o
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

V
&__inference_signature_wrapper_24445037

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
+__inference_function_with_signature_1946596a
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
Ю

/__inference_sequential_3_layer_call_fn_24445736

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445374o
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
ц
Є
/__inference_sequential_3_layer_call_fn_24445160
lambda_3_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445144o
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
_user_specified_namelambda_3_input
Є
з
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445469
lambda_3_input"
dense_6_24445463:
dense_6_24445465:
identityЂdense_6/StatefulPartitionedCallТ
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445409
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445463dense_6_24445465*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_3_input
П
F
*__inference_re_lu_1_layer_call_fn_24446037

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506h
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
Ю

/__inference_sequential_3_layer_call_fn_24445645

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445144o
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
С
G
+__inference_lambda_2_layer_call_fn_24445999

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445561h
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
ђ{
ѕ
$__inference__traced_restore_24446260
file_prefix&
assignvariableop_global_step:	 5
#assignvariableop_1_dense_6_kernel_1:/
!assignvariableop_2_dense_6_bias_1:>
$assignvariableop_3_conv2d_1_kernel_1:0
"assignvariableop_4_conv2d_1_bias_1:
yassignvariableop_5_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel:M 
wassignvariableop_6_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias: 
yassignvariableop_7_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel:  
wassignvariableop_8_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias: 
rassignvariableop_9_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel:	 
}assignvariableop_10_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel:

qassignvariableop_11_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias:	}
jassignvariableop_12_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:	v
hassignvariableop_13_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:4
"assignvariableop_14_dense_6_kernel:.
 assignvariableop_15_dense_6_bias:=
#assignvariableop_16_conv2d_1_kernel:/
!assignvariableop_17_conv2d_1_bias:t
bassignvariableop_18_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel:M n
`assignvariableop_19_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias: u
cassignvariableop_20_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel:  o
aassignvariableop_21_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias: n
[assignvariableop_22_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel:	  x
eassignvariableop_23_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel:	( h
Yassignvariableop_24_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias:	 U
Cassignvariableop_25_adversary_agent_valuernnnetwork_dense_11_kernel:(O
Aassignvariableop_26_adversary_agent_valuernnnetwork_dense_11_bias:
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
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_6_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_1_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_5AssignVariableOpyassignvariableop_5_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_6AssignVariableOpwassignvariableop_6_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_7AssignVariableOpyassignvariableop_7_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_8AssignVariableOpwassignvariableop_8_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:с
AssignVariableOp_9AssignVariableOprassignvariableop_9_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_10AssignVariableOp}assignvariableop_10_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_11AssignVariableOpqassignvariableop_11_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_12AssignVariableOpjassignvariableop_12_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_13AssignVariableOphassignvariableop_13_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_18AssignVariableOpbassignvariableop_18_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_19AssignVariableOp`assignvariableop_19_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_20AssignVariableOpcassignvariableop_20_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_21AssignVariableOpaassignvariableop_21_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_22AssignVariableOp[assignvariableop_22_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_23AssignVariableOpeassignvariableop_23_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_24AssignVariableOpYassignvariableop_24_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_25AssignVariableOpCassignvariableop_25_adversary_agent_valuernnnetwork_dense_11_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adversary_agent_valuernnnetwork_dense_11_biasIdentity_26:output:0"/device:CPU:0*
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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445194

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
ї
к
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445587

inputs+
conv2d_1_24445579:
conv2d_1_24445581:
identityЂ conv2d_1/StatefulPartitionedCallТ
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445561
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445579conv2d_1_24445581*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
Є
/__inference_sequential_2_layer_call_fn_24445795

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445587o
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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446006

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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445272

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

т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445615
lambda_2_input+
conv2d_1_24445607:
conv2d_1_24445609:
identityЂ conv2d_1/StatefulPartitionedCallЪ
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445483
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445607conv2d_1_24445609*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445355

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

т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445326
lambda_2_input+
conv2d_1_24445318:
conv2d_1_24445320:
identityЂ conv2d_1/StatefulPartitionedCallЪ
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445194
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445318conv2d_1_24445320*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445066

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

Ї
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445727

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identityЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

U
%__inference_get_initial_state_1947323

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
Э
k
+__inference_function_with_signature_1946612
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
__inference_<lambda>_2176^
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
ў
Ќ
/__inference_sequential_2_layer_call_fn_24445235
lambda_2_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445228o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
ў
Ќ
/__inference_sequential_2_layer_call_fn_24445603
lambda_2_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445587o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
Ч
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445940

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
Г
H
,__inference_flatten_3_layer_call_fn_24445934

inputs
identityВ
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225`
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
щ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446042

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
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445120

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
ф

J__inference_sequential_3_layer_call_and_return_conditional_losses_24445777

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :Ь
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџg
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445929

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
Ч
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446053

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

[
+__inference_function_with_signature_1946596

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
%__inference_get_initial_state_1946591a
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

Ї
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445711

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identityЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506

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
С
G
+__inference_lambda_2_layer_call_fn_24445994

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445483h
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

ї
+__inference_function_with_signature_1946529
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
__inference_action_1946496k
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
к
f
&__inference_signature_wrapper_24445045
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
+__inference_function_with_signature_1946612^
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
ў
Ќ
/__inference_sequential_2_layer_call_fn_24445314
lambda_2_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445298o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input

Ї
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445827

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identityЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџW
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџi
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
з
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445459
lambda_3_input"
dense_6_24445453:
dense_6_24445455:
identityЂdense_6/StatefulPartitionedCallТ
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445355
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445453dense_6_24445455*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_3_input
Ч
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514

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
Ё
G
+__inference_lambda_3_layer_call_fn_24445950

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445409`
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

Я
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445085

inputs"
dense_6_24445079:
dense_6_24445081:
identityЂdense_6/StatefulPartitionedCallК
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445066
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445079dense_6_24445081*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
Є
/__inference_sequential_3_layer_call_fn_24445381
lambda_3_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445374o
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
_user_specified_namelambda_3_input
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445409

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
Ш	
і
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078

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
ц
Є
/__inference_sequential_3_layer_call_fn_24445449
lambda_3_input
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445433o
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
_user_specified_namelambda_3_input
ї
к
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445517

inputs+
conv2d_1_24445496:
conv2d_1_24445498:
identityЂ conv2d_1/StatefulPartitionedCallТ
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445483
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445496conv2d_1_24445498*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445506ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445561

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
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445857

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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446013

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
С
G
+__inference_lambda_2_layer_call_fn_24445881

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445194h
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

Я
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445144

inputs"
dense_6_24445138:
dense_6_24445140:
identityЂdense_6/StatefulPartitionedCallК
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445120
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445138dense_6_24445140*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

/__inference_sequential_3_layer_call_fn_24445745

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445433o
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
ў
Ќ
/__inference_sequential_2_layer_call_fn_24445524
lambda_2_input!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445517o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_namelambda_2_input
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445960

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
ћ
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445970

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
ї
к
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445298

inputs+
conv2d_1_24445290:
conv2d_1_24445292:
identityЂ conv2d_1/StatefulPartitionedCallТ
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445272
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445290conv2d_1_24445292*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю
 
+__inference_conv2d_1_layer_call_fn_24446022

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallу
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495w
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
в
-
+__inference_function_with_signature_1946623у
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
__inference_<lambda>_2179*(
_construction_contextkEagerRuntime*
_input_shapes 
ц
Є
/__inference_sequential_2_layer_call_fn_24445786

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445517o
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

U
%__inference_get_initial_state_1946591

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
п

__inference_action_1946496
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
policy_state
policy_state_1
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: 
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2ЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpЂtActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpЂsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpЂrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpЂkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpЂjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpЂlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpЂVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpЂUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:џџџџџџџџџМ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџЌ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ў
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH   
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџHЃ
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџMЈ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџM   п
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0х
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ №
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
:џџџџџџџџџ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:Љ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:й
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ д
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:І
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ј
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЊ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџќ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 ќ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 з
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџл
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ч
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ъ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџф
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0э
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџІ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitџ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџв
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџљ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџе
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџд
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџі
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџй
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :к
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџђ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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
:џџџџџџџџџА

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџА

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЛ
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*В
_input_shapes 
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : 2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2ь
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2ъ
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2ш
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2к
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2и
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2м
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2А
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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445893

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
Ш	
і
E__inference_dense_6_layer_call_and_return_conditional_losses_24445876

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
ї
к
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445228

inputs+
conv2d_1_24445207:
conv2d_1_24445209:
identityЂ conv2d_1/StatefulPartitionedCallТ
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445194
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445207conv2d_1_24445209*
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
GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206у
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217ж
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
гK
Т
!__inference__traced_save_24446169
file_prefix*
&savev2_global_step_read_readvariableop	/
+savev2_dense_6_kernel_1_read_readvariableop-
)savev2_dense_6_bias_1_read_readvariableop0
,savev2_conv2d_1_kernel_1_read_readvariableop.
*savev2_conv2d_1_bias_1_read_readvariableop
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel_read_readvariableop
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias_read_readvariableop
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel_read_readvariableop
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias_read_readvariableop~
zsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel_read_readvariableop
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel_read_readvariableop|
xsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias_read_readvariableopu
qsavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableops
osavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopm
isavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel_read_readvariableopk
gsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias_read_readvariableopn
jsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel_read_readvariableopl
hsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias_read_readvariableopf
bsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel_read_readvariableopp
lsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel_read_readvariableopd
`savev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias_read_readvariableopN
Jsavev2_adversary_agent_valuernnnetwork_dense_11_kernel_read_readvariableopL
Hsavev2_adversary_agent_valuernnnetwork_dense_11_bias_read_readvariableop
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
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ў
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop+savev2_dense_6_kernel_1_read_readvariableop)savev2_dense_6_bias_1_read_readvariableop,savev2_conv2d_1_kernel_1_read_readvariableop*savev2_conv2d_1_bias_1_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias_read_readvariableopzsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel_read_readvariableopxsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias_read_readvariableopqsavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableoposavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableopisavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel_read_readvariableopgsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias_read_readvariableopjsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel_read_readvariableophsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias_read_readvariableopbsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel_read_readvariableoplsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel_read_readvariableop`savev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias_read_readvariableopJsavev2_adversary_agent_valuernnnetwork_dense_11_kernel_read_readvariableopHsavev2_adversary_agent_valuernnnetwork_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
Њ

џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445206

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
П
F
*__inference_re_lu_1_layer_call_fn_24445924

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445217h
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
Њ

џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446032

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
Ё
G
+__inference_lambda_3_layer_call_fn_24445945

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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445355`
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
п
(
&__inference_signature_wrapper_24445049ѕ
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
+__inference_function_with_signature_1946623*(
_construction_contextkEagerRuntime*
_input_shapes 

Я
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445433

inputs"
dense_6_24445427:
dense_6_24445429:
identityЂdense_6/StatefulPartitionedCallК
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445409
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445427dense_6_24445429*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

*__inference_dense_6_layer_call_fn_24445866

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallк
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445078o
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
Г
H
,__inference_flatten_3_layer_call_fn_24446047

inputs
identityВ
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
GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445514`
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
Ў
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445483

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

Я
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445374

inputs"
dense_6_24445368:
dense_6_24445370:
identityЂdense_6/StatefulPartitionedCallК
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445355
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445368dense_6_24445370*
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
GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445367w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

/__inference_sequential_3_layer_call_fn_24445636

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallп
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
GPU 2J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445085o
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
Ч
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445225

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
Њ

џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445495

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
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:њЧ
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
__inference_action_1946862
__inference_action_1947097Ч
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
#__inference_distribution_fn_1947307Ћ
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
%__inference_get_initial_state_1947323І
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
__inference_<lambda>_2179"
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
__inference_<lambda>_2176"
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
 :2dense_6/kernel
:2dense_6/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
x:vM 2fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel
r:p 2dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias
x:v  2fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel
r:p 2dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias
r:p	 2_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel
}:{
2iadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel
l:j2]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias
i:g	2Vadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
b:`2Tadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
 :2dense_6/kernel
:2dense_6/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
`:^M 2Nadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel
Z:X 2Ladversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias
a:_  2Oadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel
[:Y 2Madversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias
Z:X	  2Gadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel
d:b	( 2Qadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel
T:R 2Eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias
A:?(2/adversary_agent/ValueRnnNetwork/dense_11/kernel
;:92-adversary_agent/ValueRnnNetwork/dense_11/bias
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
__inference_action_1946862	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"Ч
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
__inference_action_1947097time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/directiontime_step/observation/image"policy_state/actor_network_state/0"policy_state/actor_network_state/1"Ч
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
#__inference_distribution_fn_1947307	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"Ћ
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
%__inference_get_initial_state_1947323
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
ХBТ
&__inference_signature_wrapper_24445028
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
аBЭ
&__inference_signature_wrapper_24445037
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
&__inference_signature_wrapper_24445045"
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
&__inference_signature_wrapper_24445049"
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
њ
щtrace_0
ъtrace_1
ыtrace_2
ьtrace_32
/__inference_sequential_3_layer_call_fn_24445092
/__inference_sequential_3_layer_call_fn_24445636
/__inference_sequential_3_layer_call_fn_24445645
/__inference_sequential_3_layer_call_fn_24445160Р
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
ц
эtrace_0
юtrace_1
яtrace_2
№trace_32ѓ
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445661
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445677
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445170
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445180Р
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
њ
trace_0
trace_1
trace_2
trace_32
/__inference_sequential_2_layer_call_fn_24445235
/__inference_sequential_2_layer_call_fn_24445686
/__inference_sequential_2_layer_call_fn_24445695
/__inference_sequential_2_layer_call_fn_24445314Р
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
ц
trace_0
trace_1
trace_2
trace_32ѓ
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445711
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445727
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445326
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445338Р
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
њ
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_32
/__inference_sequential_3_layer_call_fn_24445381
/__inference_sequential_3_layer_call_fn_24445736
/__inference_sequential_3_layer_call_fn_24445745
/__inference_sequential_3_layer_call_fn_24445449Р
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
ц
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32ѓ
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445761
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445777
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445459
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445469Р
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
њ
нtrace_0
оtrace_1
пtrace_2
рtrace_32
/__inference_sequential_2_layer_call_fn_24445524
/__inference_sequential_2_layer_call_fn_24445786
/__inference_sequential_2_layer_call_fn_24445795
/__inference_sequential_2_layer_call_fn_24445603Р
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
ц
сtrace_0
тtrace_1
уtrace_2
фtrace_32ѓ
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445811
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445827
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445615
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445627Р
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
и
љtrace_0
њtrace_12
+__inference_lambda_3_layer_call_fn_24445832
+__inference_lambda_3_layer_call_fn_24445837Р
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

ћtrace_0
ќtrace_12г
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445847
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445857Р
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
№
trace_02б
*__inference_dense_6_layer_call_fn_24445866Ђ
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

trace_02ь
E__inference_dense_6_layer_call_and_return_conditional_losses_24445876Ђ
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
B
/__inference_sequential_3_layer_call_fn_24445092lambda_3_input"Р
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
/__inference_sequential_3_layer_call_fn_24445636inputs"Р
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
/__inference_sequential_3_layer_call_fn_24445645inputs"Р
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
/__inference_sequential_3_layer_call_fn_24445160lambda_3_input"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445661inputs"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445677inputs"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445170lambda_3_input"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445180lambda_3_input"Р
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
и
trace_0
trace_12
+__inference_lambda_2_layer_call_fn_24445881
+__inference_lambda_2_layer_call_fn_24445886Р
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

trace_0
trace_12г
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445893
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445900Р
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
ё
trace_02в
+__inference_conv2d_1_layer_call_fn_24445909Ђ
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

trace_02э
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445919Ђ
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
№
trace_02б
*__inference_re_lu_1_layer_call_fn_24445924Ђ
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

trace_02ь
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445929Ђ
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
ђ
 trace_02г
,__inference_flatten_3_layer_call_fn_24445934Ђ
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

Ёtrace_02ю
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445940Ђ
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
B
/__inference_sequential_2_layer_call_fn_24445235lambda_2_input"Р
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
/__inference_sequential_2_layer_call_fn_24445686inputs"Р
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
/__inference_sequential_2_layer_call_fn_24445695inputs"Р
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
/__inference_sequential_2_layer_call_fn_24445314lambda_2_input"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445711inputs"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445727inputs"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445326lambda_2_input"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445338lambda_2_input"Р
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
и
Їtrace_0
Јtrace_12
+__inference_lambda_3_layer_call_fn_24445945
+__inference_lambda_3_layer_call_fn_24445950Р
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

Љtrace_0
Њtrace_12г
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445960
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445970Р
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
№
Аtrace_02б
*__inference_dense_6_layer_call_fn_24445979Ђ
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

Бtrace_02ь
E__inference_dense_6_layer_call_and_return_conditional_losses_24445989Ђ
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
B
/__inference_sequential_3_layer_call_fn_24445381lambda_3_input"Р
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
/__inference_sequential_3_layer_call_fn_24445736inputs"Р
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
/__inference_sequential_3_layer_call_fn_24445745inputs"Р
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
/__inference_sequential_3_layer_call_fn_24445449lambda_3_input"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445761inputs"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445777inputs"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445459lambda_3_input"Р
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445469lambda_3_input"Р
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
и
Зtrace_0
Иtrace_12
+__inference_lambda_2_layer_call_fn_24445994
+__inference_lambda_2_layer_call_fn_24445999Р
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

Йtrace_0
Кtrace_12г
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446006
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446013Р
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
ё
Рtrace_02в
+__inference_conv2d_1_layer_call_fn_24446022Ђ
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

Сtrace_02э
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446032Ђ
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
№
Чtrace_02б
*__inference_re_lu_1_layer_call_fn_24446037Ђ
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

Шtrace_02ь
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446042Ђ
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
ђ
Юtrace_02г
,__inference_flatten_3_layer_call_fn_24446047Ђ
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

Яtrace_02ю
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446053Ђ
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
B
/__inference_sequential_2_layer_call_fn_24445524lambda_2_input"Р
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
/__inference_sequential_2_layer_call_fn_24445786inputs"Р
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
/__inference_sequential_2_layer_call_fn_24445795inputs"Р
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
/__inference_sequential_2_layer_call_fn_24445603lambda_2_input"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445811inputs"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445827inputs"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445615lambda_2_input"Р
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445627lambda_2_input"Р
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
+__inference_lambda_3_layer_call_fn_24445832inputs"Р
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
+__inference_lambda_3_layer_call_fn_24445837inputs"Р
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445847inputs"Р
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445857inputs"Р
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
оBл
*__inference_dense_6_layer_call_fn_24445866inputs"Ђ
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24445876inputs"Ђ
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
+__inference_lambda_2_layer_call_fn_24445881inputs"Р
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
+__inference_lambda_2_layer_call_fn_24445886inputs"Р
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445893inputs"Р
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445900inputs"Р
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
+__inference_conv2d_1_layer_call_fn_24445909inputs"Ђ
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445919inputs"Ђ
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
*__inference_re_lu_1_layer_call_fn_24445924inputs"Ђ
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445929inputs"Ђ
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
,__inference_flatten_3_layer_call_fn_24445934inputs"Ђ
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445940inputs"Ђ
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
+__inference_lambda_3_layer_call_fn_24445945inputs"Р
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
+__inference_lambda_3_layer_call_fn_24445950inputs"Р
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445960inputs"Р
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445970inputs"Р
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
оBл
*__inference_dense_6_layer_call_fn_24445979inputs"Ђ
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24445989inputs"Ђ
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
+__inference_lambda_2_layer_call_fn_24445994inputs"Р
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
+__inference_lambda_2_layer_call_fn_24445999inputs"Р
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446006inputs"Р
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446013inputs"Р
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
+__inference_conv2d_1_layer_call_fn_24446022inputs"Ђ
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446032inputs"Ђ
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
*__inference_re_lu_1_layer_call_fn_24446037inputs"Ђ
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446042inputs"Ђ
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
,__inference_flatten_3_layer_call_fn_24446047inputs"Ђ
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446053inputs"Ђ
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
__inference_<lambda>_2176Ђ

Ђ 
Њ " 	1
__inference_<lambda>_2179Ђ

Ђ 
Њ "Њ ж
__inference_action_1946862ЗПЂЛ
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
__inference_action_1947097Ђ
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
infoЂ Ж
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445919l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Ж
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446032l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_1_layer_call_fn_24445909_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
+__inference_conv2d_1_layer_call_fn_24446022_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЅ
E__inference_dense_6_layer_call_and_return_conditional_losses_24445876\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ѕ
E__inference_dense_6_layer_call_and_return_conditional_losses_24445989\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_6_layer_call_fn_24445866O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ}
*__inference_dense_6_layer_call_fn_24445979O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЖ
#__inference_distribution_fn_1947307ЛЂЗ
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
infoЂ Ћ
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445940`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџH
 Ћ
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446053`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџH
 
,__inference_flatten_3_layer_call_fn_24445934S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџH
,__inference_flatten_3_layer_call_fn_24446047S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџHе
%__inference_get_initial_state_1947323Ћ"Ђ
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445893p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445900p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446006p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 К
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446013p?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_lambda_2_layer_call_fn_24445881c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
+__inference_lambda_2_layer_call_fn_24445886c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџ
+__inference_lambda_2_layer_call_fn_24445994c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ
+__inference_lambda_2_layer_call_fn_24445999c?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџЊ
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445847`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ
 Њ
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445857`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ
 Њ
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445960`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ
 Њ
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445970`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_lambda_3_layer_call_fn_24445832S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ
+__inference_lambda_3_layer_call_fn_24445837S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџ
+__inference_lambda_3_layer_call_fn_24445945S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџ
+__inference_lambda_3_layer_call_fn_24445950S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџБ
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445929h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Б
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446042h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
*__inference_re_lu_1_layer_call_fn_24445924[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ
*__inference_re_lu_1_layer_call_fn_24446037[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџТ
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445326tGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 Т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445338tGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 Т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445615tGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 Т
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445627tGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 К
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445711l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 К
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445727l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 К
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445811l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџH
 К
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445827l?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџH
 
/__inference_sequential_2_layer_call_fn_24445235gGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445314gGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445524gGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445603gGЂD
=Ђ:
0-
lambda_2_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445686_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445695_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445786_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџH
/__inference_sequential_2_layer_call_fn_24445795_?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџHК
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445170l?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 К
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445180l?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 К
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445459l?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 К
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445469l?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 В
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445661d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 В
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445677d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 В
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445761d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 В
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445777d7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
/__inference_sequential_3_layer_call_fn_24445092_?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445160_?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445381_?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445449_?Ђ<
5Ђ2
(%
lambda_3_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445636W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445645W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445736W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_3_layer_call_fn_24445745W7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџђ
&__inference_signature_wrapper_24445028ЧиЂд
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
state/actor_network_state/1џџџџџџџџџњ
&__inference_signature_wrapper_24445037Я0Ђ-
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
&__inference_signature_wrapper_244450450Ђ

Ђ 
Њ "Њ

int64
int64 	>
&__inference_signature_wrapper_24445049Ђ

Ђ 
Њ "Њ 