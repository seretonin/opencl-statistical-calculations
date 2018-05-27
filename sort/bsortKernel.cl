#define CHALLENGE_LEN 32

typedef struct entry_s{
    char challenge[CHALLENGE_LEN+1];
    int  idx;
}entry_t;

int compare(char *p1, char *p2, int len)
{
    for(int i = 0; i < len; i++)
    {
        if(p1[i] == p2[i]){
            continue;
        }else if(p1[i] < p2[i]){
            return -1;
        }else{
            return 1;
        }
    }
    return 0;
}

__kernel
void parallelBitonicSort(__global float * data,
                         const uint stage,
                         const uint subStage,
                         const uint direction) {

    uint threadId = get_global_id(0);
    uint sortIncreasing = direction;

    uint distanceBetweenPairs = 1 << (stage - subStage);
    uint blockWidth   = distanceBetweenPairs << 1;

    uint leftId = (threadId % distanceBetweenPairs) + (threadId / distanceBetweenPairs) * blockWidth;

    uint rightId = leftId + distanceBetweenPairs;

    float leftElement = data[leftId];
    float rightElement = data[rightId];

    uint sameDirectionBlockWidth = 1 << stage;

    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    float greater;
    float lesser;
    if(leftElement > rightElement) {
        greater = leftElement;
        lesser  = rightElement;
    } else {
        greater = rightElement;
        lesser  = leftElement;
    }

    if(sortIncreasing) {
        data[leftId]  = lesser;
        data[rightId] = greater;
    } else {
        data[leftId]  = greater;
        data[rightId] = lesser;
    }
}

