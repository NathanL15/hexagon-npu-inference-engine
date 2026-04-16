#include <iostream>
#include <windows.h>

int main() {
    std::cout << "Starting Native ARM64 Engine..." << std::endl;

    // Define the path to the Qualcomm CPU Backend DLL
    // Note: When you run the .exe, Windows needs to be able to find this file.
    // For now, we will use a relative path assuming you run from the build folder.
    LPCSTR dllPath = "./third_party/qnn_sdk/lib/aarch64-windows-msvc/QnnCpu.dll";

    // Attempt to load the library into memory
    HMODULE hQnnLib = LoadLibraryA(dllPath);

    if (hQnnLib != NULL) {
        std::cout << "[SUCCESS] Successfully loaded Qualcomm CPU Backend (QnnCpu.dll)!" << std::endl;
        FreeLibrary(hQnnLib);
    } else {
        std::cerr << "[FAILED] Could not load the DLL. Error Code: " << GetLastError() << std::endl;
        std::cerr << "Make sure the path to the DLL is correct relative to where you are running the program." << std::endl;
    }

    return 0;
}
