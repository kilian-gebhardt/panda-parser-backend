//
// Created by kilian on 13/03/17.
//

#ifndef DEBUG_UTIL_H
#define DEBUG_UTIL_H

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

// from https://stackoverflow.com/a/77336
void handler(int sig) {
  void *array[20];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 20);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

// signal(SIGSEGV, handler);   // install the handler
// signal(SIGABRT, handler);


#endif //DEBUG_UTIL_H
