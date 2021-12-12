#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

template<typename T>
void print(T value) {
    // hail python!
    std::cout << value << std::endl;
}
template<typename T1, typename... T>
void print(T1 first, T... other) {
    std::cout << first << " ";
    print(other...);
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

#define release_for(TYPE, RELEASE_FUNCTION)                                                                            \
    void release(TYPE to_release) {                                                                                    \
        OCL_SAFE_CALL(RELEASE_FUNCTION(to_release));                                                                   \
    }

release_for(cl_command_queue, clReleaseCommandQueue);
release_for(cl_mem, clReleaseMemObject);
release_for(cl_context, clReleaseContext);
release_for(cl_sampler, clReleaseSampler);
release_for(cl_program, clReleaseProgram);
release_for(cl_kernel, clReleaseKernel);
release_for(cl_event, clReleaseEvent);

class Releaser{
    class ObjectToReleaseParent{
    public:
        virtual ~ObjectToReleaseParent(){};
    };

    template<typename T>
    class ObjectToRelease : public ObjectToReleaseParent{
        T &obj;
    public:
        ObjectToRelease(T &obj) : obj(obj) {}
        ~ObjectToRelease() {
            release(obj);
        }
    };

    std::vector<ObjectToReleaseParent *> objects_to_release;

public:
    Releaser(){}

    template<typename T>
    Releaser(T &obj_to_release){
        add(obj_to_release);
    }

    template<typename T>
    void add(T &obj_to_release) {
        auto object_p = new ObjectToRelease<T>(obj_to_release);
        objects_to_release.push_back(object_p);
    }

    template<typename T0, typename... T>
    void add(T0 &first, T&... other) {
        add(first);
        add(other...);
    }

    ~Releaser(){
        for (auto obj_to_release:objects_to_release)
            delete obj_to_release;
    }
};


void compile_program(cl_program program, cl_device_id device) {
    size_t log_size = 0;
    size_t errcode = clBuildProgram(program, 1, &device, "", NULL, NULL);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1)
        print("Log:\n\t", log.data());
    OCL_SAFE_CALL(errcode);
}

int main() {
    Releaser releaser;
    // TODO choose device by its property (type - GPU)
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    cl_platform_id platform = platforms[0];

    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::vector<cl_device_id> devices(devicesCount);
    print("Num of devices:", devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    cl_device_id device = devices[0];

    // Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (platform), 0};
    cl_int errcode;
    cl_context context = clCreateContext(props, devicesCount, devices.data(), NULL, NULL, &errcode);
    OCL_SAFE_CALL(errcode);

    releaser.add(context);

    // Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode);
    OCL_SAFE_CALL(errcode);
    releaser.add(queue);

    unsigned int n = 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    const size_t arrSize = sizeof(float) * n;
    cl_mem as_gpu = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, arrSize, as.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem bs_gpu = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, arrSize, bs.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem cs_gpu = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, arrSize, cs.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    releaser.add(as_gpu, bs_gpu, cs_gpu);

    // td 6 Выполните td 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // print(kernel_sources);
    }

    // Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char *kernel_str_pointer = kernel_sources.c_str();
    size_t len = kernel_sources.length();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_str_pointer, &len, &errcode);
    OCL_SAFE_CALL(errcode);
    releaser.add(program);

    // Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    compile_program(program, device);


    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела

    // Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode);
    OCL_SAFE_CALL(errcode);
    releaser.add(kernel);

    // Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
    }

    // Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        cl_event event;
        Releaser r(event);
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(
                    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &workGroupSize, 0, NULL, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / (1000000000) << std::endl;

        // Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * arrSize / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            Releaser r(event);
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cs_gpu, CL_TRUE, 0, arrSize, cs.data(), 0, NULL, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << arrSize / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
    print("All data is correct");

    return 0;
}
