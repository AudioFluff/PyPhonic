# 10 July 2024

New VST release:
* Plugin will now detect and stop local Python code if it's in an infinite loop (`process` function takes longer than 2s to return) instead of grinding and killing the DAW. It can be restarted.

# 9 July 2024

New VST release:
* Fixed MIDI note deallocation issue (thanks PW)
* Minor performance enhancements

# 8 July 2024

New VST release:
* Added ASIO support to the standalone executable (thanks RD)
* Fixed crash during plugin scan in some DAW hosts
* Added notification when preset download can't work due to lack of internet