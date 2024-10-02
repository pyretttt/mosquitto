#include "quick_sort.h"

// static void swap(triangle_t *a, triangle_t *b) {
//   triangle_t t = *a;
//   *a = *b;
//   *b = t;
// }

// static int partition(triangle_t array[], int low, int high) {
  
//   float pivot = array[high].avg_depth;
  
//   int i = low;

//   for (int j = low; j < high; j++) {
//     if (array[j].avg_depth <= pivot) {
//       if (i != j) swap(&array[i], &array[j]);
//       i++;
//     }
//   }

//   swap(&array[i], &array[high]);
  
//   return i;
// }

// void quickSort(triangle_t array[], int low, int high) {
//   if (low < high) {
    
//     int pi = partition(array, low, high);
//     quickSort(array, low, pi - 1);
//     quickSort(array, pi + 1, high);
//   }
// }