#include <Eigen/Core>
#include <stdexcept>
#include <iostream>
#include <LBFGS.h>

using namespace LBFGSpp;

using Scalar = double;
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
        if (!std::isfinite(fx))
        {
            throw std::runtime_error("fx is not finite");
        }
        return fx;
    }
};

int main()
{
    LBFGSParam<Scalar> param;
    LBFGSSolver<Scalar, LineSearchBracketing> solver(param);

    for (int n = 2; n <= 16; n += 2)
    {
        std::cout << "n = " << n << std::endl;
        Rosenbrock fun(n);
        for (int test = 0; test < 1024; test++)
        {
            Vector x = Vector::Random(n);
            Scalar fx;
            int niter = solver.minimize(fun, x, fx);

            Scalar diff = (x.array() - 1.0).abs().maxCoeff();
            if (diff > 1e-4)
            {
                throw std::runtime_error("Error is larger than 1e-4");
            }
        }
        std::cout << "Test passed!" << std::endl << std::endl;
    }

    return 0;
}
