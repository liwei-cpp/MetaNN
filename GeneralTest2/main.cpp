#include <data/cardinal/test_cardinal_pack.h>
#include <data/batch/test_batch_pack.h>
#include <data/sequence/test_sequence_pack.h>

int main()
{
	Test::Data::Cardinal::test_cardinal_pack();
    Test::Data::Batch::test_batch_pack();
    Test::Data::Sequence::test_sequence_pack();
	return 0;
}
