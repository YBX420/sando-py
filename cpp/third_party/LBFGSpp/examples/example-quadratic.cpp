#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>

using namespace LBFGSpp;

using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

Scalar quadratic(const Vector& x, Vector& grad)
{
    const int n = x.size();
    Vector d(n);
    for (int i = 0; i < n; i++)
        d[i] = i;

    Scalar f = (x - d).squaredNorm();
    grad.noalias() = 2.0 * (x - d);
    return f;
}

int main()
{
    const int n = 10;
    LBFGSParam<Scalar> param;
    LBFGSSolver<Scalar> solver(param);

    Vector x = Vector::Zero(n);
    Scalar fx;
    int niter = solver.minimize(quadratic, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    std::cout << "grad norm = " << solver.final_grad_norm() << std::endl;

    return 0;
}
