#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
void setTime();
double getTime();
#ifdef __cplusplus
}
#endif
#endif	//	SIMPLETIMER_H

enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SmemMode {UseSmem, UseNoSmem};
