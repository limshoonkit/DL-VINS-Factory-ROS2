#include "../../include/backend/backend_interface.hpp"
#include "../../include/backend/ceres_backend.hpp"
#include <rclcpp/logging.hpp>

namespace uosm
{
    namespace perception
    {

        std::unique_ptr<IBackendSolver> BackendSolverFactory::create(
            BackendType type,
            const BackendSolverConfig &config)
        {
            switch (type)
            {
            case BackendType::CERES:
                return std::make_unique<CeresBackend>(config);

            case BackendType::HYPERION:
                // TODO: Implement Hyperion backend
                RCLCPP_WARN(rclcpp::get_logger("backend_factory"), "Hyperion backend not implemented, using Ceres");
                return std::make_unique<CeresBackend>(config);
            }
            return std::make_unique<CeresBackend>(config);
        }

    } // namespace perception
} // namespace uosm