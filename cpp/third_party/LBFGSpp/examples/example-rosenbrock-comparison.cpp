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
    std::size_t ncalls;

public:
    Rosenbrock(int n_) : n(n_), ncalls(0) {}
    Scalar operator()(const Vector& x, Vector& grad)
    {
//        std::cout << x << std::endl;
        ncalls += 1;

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

    const std::size_t get_ncalls()
    {
        return ncalls;
    }
};

inline void validate_solution(const Vector& x)
{
    Scalar diff = (x.array() - 1.0).abs().maxCoeff();
    if (diff > 1e-4)
    {
        throw std::runtime_error("Error is larger than 1e-4");
    }
}

int main()
{
    LBFGSParam<Scalar> param;
    param.    linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    param.max_linesearch = 256;

    LBFGSSolver<Scalar, LineSearchBacktracking > solver_backtrack(param);
    LBFGSSolver<Scalar, LineSearchBracketing   > solver_bracket  (param);
    LBFGSSolver<Scalar, LineSearchNocedalWright> solver_nocedal  (param);
    LBFGSSolver<Scalar, LineSearchMoreThuente>   solver_more     (param);

    const int tests_per_n = 1024;

    for (int n = 2; n <= 24; n += 2)
    {
        std::cout << "n = " << n << std::endl;
        Rosenbrock fun_backtrack(n),
                   fun_bracket  (n),
                   fun_nocedal  (n),
                   fun_more     (n);
        int niter_backtrack = 0,
            niter_bracket   = 0,
            niter_nocedal   = 0,
            niter_more      = 0;
        for (int test = 0; test < tests_per_n; test++)
        {
            Vector x, x0 = Vector::Random(n);
            Scalar fx;

            x = x0; niter_backtrack += solver_backtrack.minimize(fun_backtrack, x, fx); validate_solution(x);
            x = x0; niter_bracket   += solver_bracket  .minimize(fun_bracket  , x, fx); validate_solution(x);
            x = x0; niter_nocedal   += solver_nocedal  .minimize(fun_nocedal  , x, fx); validate_solution(x);
            x = x0; niter_more      += solver_more     .minimize(fun_more     , x, fx); validate_solution(x);
        }
        std::cout << "  Average #calls:" << std::endl;
        std::cout << "  LineSearchBacktracking : " << (fun_backtrack.get_ncalls() / tests_per_n) << " calls, " << (niter_backtrack / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchBracketing   : " << (fun_bracket  .get_ncalls() / tests_per_n) << " calls, " << (niter_bracket   / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchNocedalWright: " << (fun_nocedal  .get_ncalls() / tests_per_n) << " calls, " << (niter_nocedal   / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchMoreThuente  : " << (fun_more     .get_ncalls() / tests_per_n) << " calls, " << (niter_more      / tests_per_n) << " iterations" << std::endl;
    }

    return 0;
}
