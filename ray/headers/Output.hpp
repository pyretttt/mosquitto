#include <ostream>

#include "MathUtils.hpp"

std::ostream& operator<<(std::ostream& os, ml::Vector3f const &vec)
{
      os << vec.x << " " << vec.y << " " << vec.z;
      return os;
}

std::ostream& operator<<(std::ostream& os, ml::Vector4f const &vec)
{
      os << vec.x << " " << vec.y << " " << vec.z << " " << vec.w;
      return os;
}