# 14 July 2024

* It won't die now if you unload the plugin/exit the DAW while Python is still running (thanks PW)
* "Run Python" button now accurately reflects whether Python is running or not if you close + reopen the UI.
* Fixed issues with multiple Pyphonic instances running at the same time, if one crashed or was in an infinite loop, or was deleted. Now the remaining instances continue to work.
* Performance & efficiency enhancements

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