#include "my_pendulum/my_pendulum.h"
#include "my_pendulum/operations_generic.h"


template <typename T, typename TI>
using ENVIRONMENT_FACTORY = MyPendulum<MyPendulumSpecification<T, TI, MyPendulumParameters<T>>>;