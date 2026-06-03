#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>

using namespace LBFGSpp;

using Scalar = float;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// f(x) = (x[0] - 1)^2 + 100 * (x[1] - x[0]^2)^2 +
//        (x[2] - 1)^2 + 100 * (x[3] - x[2]^2)^2 + ...
class Rosenbrock
{
private:
    int n;
public:
    Rosenbrock(int n_) : n(n_) {}
    Scalar operator()(const Vector& x, Vector& grad)
    {
        Scalar fx = 0.0;
        for (int i = 0; i < n; i += 2)
        {
            Scalar t1 = 1.0 - x[i];
            Scalar t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;
    }
};

int main()
{
    const int n = 10;
    LBFGSParam<Scalar> param;
    LBFGSSolver<Scalar> solver(param);
    Rosenbrock fun(n);

    Vector x = Vector::Zero(n);
    Scalar fx;
    int niter = solver.minimize(fun, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    std::cout << "grad = " << solver.final_grad().transpose() << std::endl;
    std::cout << "||grad|| = " << solver.final_grad_norm() << std::endl;
    std::cout << "approx_hess = \n" << solver.final_approx_hessian() << std::endl;
    std::cout << "approx_inv_hess = \n" << solver.final_approx_inverse_hessian() << std::endl;

    return 0;
}
