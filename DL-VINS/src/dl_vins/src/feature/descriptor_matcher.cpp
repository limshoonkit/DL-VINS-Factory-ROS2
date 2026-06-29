#include "../include/feature/descriptor_matcher.hpp"
#include "../include/feature/lightglue_matcher.hpp"

#include <rclcpp/logging.hpp>

namespace uosm
{
    namespace perception
    {

        std::unique_ptr<DescriptorMatcher> makeDescriptorMatcher(const std::string &type,
                                                                 const DescriptorMatcherParams &params)
        {
            if (type == "lightglue")
            {
                return std::make_unique<LightGlueMatcher>(params);
            }

            RCLCPP_WARN(rclcpp::get_logger("descriptor_matcher"),
                        "makeDescriptorMatcher: unknown matcher type '%s'. "
                        "Supported: lightglue.",
                        type.c_str());
            return nullptr;
        }

    } // namespace perception
} // namespace uosm