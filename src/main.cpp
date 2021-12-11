#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <map>
#include <string>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;
    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    // error message is not shown on popup
    std::cerr << message << std::endl;
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    for (cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;

        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t vendorNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
        std::vector<unsigned char> vendorName(vendorNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
        std::cout << "\tVendor: " << vendorName.data() << std::endl;

        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "\t" << devicesCount << ": " << std::endl;
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        size_t param_size;
        std::map<int, std::string> types({
                {CL_DEVICE_TYPE_CPU, "cpu"},
                {CL_DEVICE_TYPE_GPU, "gpu"},
                {CL_DEVICE_TYPE_ACCELERATOR, "accelerator"},
                {CL_DEVICE_TYPE_DEFAULT, "default"},
        });
        for (cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            auto device = devices[deviceIndex];
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &param_size));
            std::vector<char> deviceName(param_size);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, param_size, deviceName.data(), nullptr));

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &param_size));
            cl_ulong mem_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, param_size, &mem_size, nullptr));

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &param_size));
            cl_device_type type = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, param_size, &type, nullptr));
            std::cout << "\t\t" << deviceName.data() << " - " << types[type] << "; memory size: " << mem_size/(1024*1024) << " mb" << '\n';

        }
    }

    return 0;
}